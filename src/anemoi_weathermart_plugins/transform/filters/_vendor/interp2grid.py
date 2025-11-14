# -----------------------------------------------------------------------------
# DISCLAIMER
# -----------------------------------------------------------------------------
# Copied from:
# https://service.meteoswiss.ch/git/rasterdatabase/gridefix/gridefix-process
# while waiting for the open-source version of the API.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Imports and typing
# -----------------------------------------------------------------------------
import json
from collections.abc import Hashable
from functools import wraps
from logging import getLogger
from math import ceil
from math import prod
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional
from typing import TypeVar
from typing import Union
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.spatial import KDTree

# -----------------------------------------------------------------------------
# Logging and generic type variables
# -----------------------------------------------------------------------------
logger = getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])

# -----------------------------------------------------------------------------
# Constants and type aliases
# -----------------------------------------------------------------------------
_InterpMethods = Union[
    Literal[
        "idw",
        "temp",
        "temp_fixgrad",
        "nearest_alt",
        "snowlim",
        "nearest_3D",
        "linear_3D",
        "idw_3D",
        "upscaled",
        "upscaled_3D",
    ],
    Literal["nearest", "linear", "cubic"],
]
HEIGHT_METHODS = (
    "temp",
    "temp_fixgrad",
    "snowlim",
    "nearest_alt",
    "idw_3D",
    "nearest_3D",
    "linear_3D",
    "upscaled_3D",
)
FLAT_DIM = ["cell", "points"]
LATLON = (("latitude", "longitude"), ("lat", "lon"))
SRC_ATTRS_KEEP = ("source", "title", "institution", "history", "references", "comment")
YX = tuple(zip(["y", "latitude", "lat"], ["x", "longitude", "lon"]))
SWISS_CRS = ("EPSG:2056", "EPSG:21781")
SURF_ALT = ["surface_altitude", "HSURF", "altitude"]
REFTIME = ["forecast_reference_time", "ref_time", "reftime"]
VALIDTIME = ["time", "valid_time"]
LEADTIME = ["lead_time", "step"]
ENS_DIM = ["realization", "eps", "epsd_1", "epsm", "number"]
POI_ALT_COL = ["height_masl", "height", "altitude", "elevation"]
POI_DIM = "poi"
POI_COORD_COL = "name"


# -----------------------------------------------------------------------------
# Simple utilities
# -----------------------------------------------------------------------------
def cast_dtype(ds_out: xr.Dataset, ds_in: xr.Dataset) -> None:
    # Restore original dtypes after interpolation
    for varname, da in ds_out.items():
        dtype_in = ds_in[varname].dtype
        if da.dtype != dtype_in:
            ds_out[varname] = da.astype(dtype_in)


def memoize(maxsize: int = 10) -> Callable[[F], F]:
    # Lightweight LRU cache keyed explicitly by 'hkey'
    def real_memoize(func: F) -> F:
        cache: dict[Hashable, Any] = {}
        hkeys: list[Hashable] = []

        @wraps(func)
        def _memoize(*args: Any, **kwargs: Any) -> Any:
            hkey = kwargs.pop("hkey", None)
            if hkey in cache:
                return cache[hkey]
            result = func(*args, **kwargs)
            if hkey is not None:
                cache[hkey] = result
                hkeys.append(hkey)
                if len(hkeys) > maxsize:
                    cache.pop(hkeys.pop(0))
            return result

        return cast(F, _memoize)

    return real_memoize


# -----------------------------------------------------------------------------
# Monotonicity helpers
# -----------------------------------------------------------------------------
def is_decreasing1d(coords: xr.DataArray) -> bool:
    # True if first diff < 0
    return bool(np.ediff1d(coords.values[:2])[0] < 0)


def is_decreasing(coords: xr.DataArray, dim: str) -> bool:
    # Check order along dim (supports 2D coords)
    if coords.ndim > 1:
        other_dim = next(d for d in coords.dims if d != dim)
        mid_index = coords.sizes[other_dim] // 2
        return is_decreasing1d(coords.isel({other_dim: mid_index}))
    return is_decreasing1d(coords)


# -----------------------------------------------------------------------------
# Distance weighting and pointwise interpolation kernels
# -----------------------------------------------------------------------------
def _compute_dist_weight(dist: np.ndarray, power: float, dtype: np.dtype) -> np.ndarray:
    # Inverse distance weights normalized to 1
    increm = np.mean(dist, axis=1, keepdims=True) / 50
    w = 1 / np.power(dist + increm, power)
    w = w.astype(dtype, copy=False)
    return w / np.sum(w, axis=1, keepdims=True)


def _interp_nearest_alt(
    coords_flat: np.ndarray,
    data_flat: np.ndarray,
    pois_coords: np.ndarray,
    r_search: float = 2,
    z_weight: int = 500,
) -> np.ndarray:
    # Altitude-weighted nearest interpolation (vertical preference)
    ind = _grid_assignation(coords_flat, pois_coords, r_search, z_weight)
    return data_flat[ind,]


def _grid_assignation(
    coords_flat: np.ndarray,
    pois_coords: np.ndarray,
    r_search: float,
    z_weight: int,
    deltah_min: int = 50,
) -> np.ndarray:
    # Choose index minimizing horizontal + weighted vertical distance
    k = round(np.pi * r_search**2)
    dist_nn_horz, ind_nn = _nn_lookup(coords_flat[:, :2], pois_coords[:, :2], k)
    dist_nn_vert = np.abs(coords_flat[ind_nn, -1] - pois_coords[:, [-1]])
    dist_weighted = dist_nn_horz + z_weight * (dist_nn_vert - deltah_min).clip(0)
    return ind_nn[np.arange(ind_nn.shape[0]), np.argmin(dist_weighted, axis=1)]


def _interp_nearest(
    coords_flat: np.ndarray, data_flat: np.ndarray, pois_coords: np.ndarray
) -> np.ndarray:
    # Plain nearest neighbor
    _, ind_nn = _nn_lookup(coords_flat, pois_coords, 1)
    return data_flat[ind_nn]


def _interp_idw(
    coords_flat: np.ndarray,
    data_flat: np.ndarray,
    pois_coords: np.ndarray,
    k: int = 4,
    weight_power: float = 1.0,
) -> np.ndarray:
    # Inverse-distance weighted interpolation
    dist_nn, ind_nn = _nn_lookup(coords_flat, pois_coords, k)
    dist_weight = _compute_dist_weight(dist_nn, weight_power, data_flat.dtype)
    ndim_trail = data_flat.ndim - 1
    return np.sum(
        data_flat[ind_nn,] * dist_weight.reshape(dist_weight.shape + (1,) * ndim_trail),
        axis=1,
    )


# -----------------------------------------------------------------------------
# Core public API
# -----------------------------------------------------------------------------
def interp2grid(
    data: xr.Dataset,
    dst_grid: xr.Dataset,
    method: _InterpMethods,
    try_reg: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
    # Grid-to-grid interpolation dispatcher
    def _array2ds(data_arr: xr.DataArray) -> xr.Dataset:
        da = xr.DataArray(
            interp_data.reshape(
                (
                    dst_grid.sizes[dst_spatial_dims[0]],
                    dst_grid.sizes[dst_spatial_dims[1]],
                )
                + data_arr.shape[len(src_spatial_dims) :]
            ),
            dims=dst_spatial_dims + data_arr.dims[len(src_spatial_dims) :],
            coords={
                **data_arr.isel({dim: 0 for dim in src_spatial_dims}, drop=True).coords,
                **dst_grid.coords,
            },
            attrs=_combine_attrs(data_arr.attrs, dst_grid.attrs),
        )
        return _da_to_ds(da, vars_attrs)

    src_spatial_dims = _get_spatial_dims(data)
    dst_spatial_dims = _get_spatial_dims(dst_grid)
    data = _set_grid_mapping(data)
    dst_grid = _set_grid_mapping(dst_grid)
    dst_grid = dst_grid.drop_dims(set(dst_grid.dims) - set(dst_spatial_dims))
    cond_reg = (
        method in ("nearest", "linear")
        and len(src_spatial_dims) > 1
        and set(src_spatial_dims).issubset(data.coords)
    )
    if try_reg and cond_reg:
        return _interp_grid_regular(
            data,
            src_spatial_dims,
            dst_grid,
            dst_spatial_dims,
            method,  # type: ignore[arg-type]
        )
    dst_grid = dst_grid.transpose(*dst_spatial_dims)
    data = _slice_coarsen(
        data, dst_grid, method, src_spatial_dims, dst_spatial_dims, **kwargs
    )
    data_arr, vars_attrs = _ds_to_da(data)  # type: ignore[assignment]
    data_arr = data_arr.transpose(*src_spatial_dims, ...)
    coords_flat, data_flat, dst_coords_flat = _flatten_concat_grids(
        data_arr,
        src_spatial_dims,
        dst_grid,
        dst_spatial_dims,
        method,  # type: ignore[arg-type]
    )
    if method in INTERP_FUNC:
        interp_data = INTERP_FUNC[method](
            coords_flat, data_flat, dst_coords_flat, **kwargs
        )  # type: ignore[index]
    elif method in ("linear", "linear_3D", "cubic"):
        interp_data = griddata(
            coords_flat,
            data_flat,
            dst_coords_flat,
            method.rsplit("_")[0],
        ).astype(data_flat.dtype, copy=False)
    else:
        raise ValueError(f"Unknown interpolation method {method!r}")
    return _array2ds(data_arr)  # type: ignore[arg-type]


def _interp_grid_regular(
    data: xr.Dataset,
    src_spatial_dims: tuple[str, str],
    dst_grid: xr.Dataset,
    dst_spatial_dims: tuple[str, str],
    method: Literal["nearest", "linear"] = "linear",
) -> xr.Dataset:
    # Fast path for regular→regular (optional coarsen + xarray.interp)
    def _cond_dst_1d() -> bool:
        data_trail_size = prod(
            size for dim, size in data.sizes.items() if dim not in src_spatial_dims
        )
        dst_sizes = dst_grid.sizes
        dim_ratio = (
            dst_sizes[dst_spatial_dims[1]] * dst_sizes[dst_spatial_dims[0]]
        ) / data_trail_size
        return (dim_ratio > 2000 or method == "linear") and set(
            dst_spatial_dims
        ).issubset(dst_grid.coords)

    def _reproject_dst_2d() -> tuple[xr.DataArray, xr.DataArray]:
        x_vals = dst_grid[dst_coords_crs[1]].values
        y_vals = dst_grid[dst_coords_crs[0]].values
        if x_vals.ndim == 1:
            x_vals, y_vals = np.meshgrid(x_vals, y_vals)
        x_vals, y_vals = _reproject_grid(x_vals, y_vals, dst_coords_crs[2], src_crs)
        return xr.DataArray(x_vals, dims=dst_spatial_dims), xr.DataArray(
            y_vals, dims=dst_spatial_dims
        )

    def _coarsen_interp() -> xr.Dataset:
        src_coords_crs = src_spatial_dims + (src_crs,)
        rx, ry = _get_coarsen_ratio(
            data,
            src_spatial_dims,
            src_coords_crs,
            dst_grid,
            dst_spatial_dims,
            dst_coords_crs,
        )
        interp_data = data
        if (rx > 1) or (ry > 1):
            specs = {
                d: r
                for d, r in ((src_spatial_dims[0], ry), (src_spatial_dims[1], rx))
                if r > 1
            }
            interp_data = interp_data.coarsen(specs, boundary="trim").mean(skipna=False)  # type: ignore[attr-defined]
        interp_data = _y_increasing(interp_data, src_spatial_dims[0])
        interp_data = interp_data.interp(
            {
                src_spatial_dims[1]: x_coords,
                src_spatial_dims[0]: y_coords,
            },
            method=method,
            assume_sorted=True,
        ).drop_vars(src_spatial_dims)
        cast_dtype(interp_data, data)
        interp_data.attrs = _combine_attrs(data.attrs, dst_grid.attrs)
        return interp_data.assign_coords(dst_grid.coords)

    data = _drop_coords_reg(data, src_spatial_dims)
    src_crs = _get_grid_crs(data)
    if _cond_dst_1d():
        dst_crs = _get_grid_crs(dst_grid)
        if src_crs == dst_crs:
            dst_coords_crs = dst_spatial_dims + (dst_crs,)
            x_coords = dst_grid[dst_spatial_dims[1]].drop_vars(dst_spatial_dims[1])
            y_coords = dst_grid[dst_spatial_dims[0]].drop_vars(dst_spatial_dims[0])
            return _coarsen_interp()
    dst_grid = dst_grid.transpose(*dst_spatial_dims)
    dst_coords_crs = _get_coords_name_crs(dst_grid, dst_spatial_dims)
    x_coords, y_coords = _reproject_dst_2d()
    return _coarsen_interp()


# -----------------------------------------------------------------------------
# CRS / coordinate and grid helpers (first block)
# -----------------------------------------------------------------------------
def _drop_coords_reg(data: xr.Dataset, spatial_dims: tuple[str, str]) -> xr.Dataset:
    notdim_coords = set(data.coords) - set(data.dims)
    spd = set(spatial_dims)
    to_drop = [coord for coord in notdim_coords if spd.issubset(data[coord].dims)]

    return data.drop_vars(to_drop)


def _y_increasing(ds: xr.Dataset, y_dim: str) -> xr.Dataset:
    if _is_decreasing1d(ds[y_dim]):
        ds = ds.isel({y_dim: slice(None, None, -1)})

    return ds


def _is_decreasing1d(coords: xr.DataArray) -> bool:
    return bool(np.ediff1d(coords.values[:2])[0] < 0)


def get_latlon_coords(
    grid: xr.Dataset | xr.DataArray, nondim: bool = True
) -> tuple[str, str] | None:
    coords = tuple(grid.coords)
    if nondim:
        coords = tuple(set(coords) - set(grid.dims))
    return _get_latlon(grid, coords)  # type: ignore[arg-type]


def get_coords_name_crs(
    grid: xr.Dataset | xr.DataArray,
    spatial_dims: tuple[str, ...] | None = None,
) -> tuple[str, str, CRS]:
    latlon = get_latlon_coords(grid)
    if latlon is not None:
        grid_y, grid_x = latlon
        return grid_y, grid_x, CRS.from_string("EPSG:4326")

    if spatial_dims is None:
        spatial_dims = _get_spatial_dims(grid)

    if len(spatial_dims) == 1:
        grid_y, grid_x = _get_spatial_coords(grid)
        return grid_y, grid_x, _get_grid_crs(grid)

    grid_y, grid_x = spatial_dims
    if grid_y in grid.coords and grid_x in grid.coords:
        return grid_y, grid_x, _get_grid_crs(grid)

    raise ValueError("Missing spatial coordinates")


def _get_latlon(
    grid: xr.Dataset | xr.DataArray, dim_or_coord: tuple[str, ...]
) -> tuple[str, str] | None:
    for lat, lon in LATLON:
        if lat in dim_or_coord and lon in dim_or_coord:
            return lat, lon
    latlon: list[str | None] = [None, None]
    for name in dim_or_coord:
        coord_attrs = grid[name].attrs
        if (
            coord_attrs.get("standard_name", "") == "latitude"
            or coord_attrs.get("long_name", "") == "latitude"
        ):
            latlon[0] = name
        elif (
            coord_attrs.get("standard_name", "") == "longitude"
            or coord_attrs.get("long_name", "") == "longitude"
        ):
            latlon[1] = name

        if None not in latlon:
            break

    if None in latlon:
        return None

    return (latlon[0], latlon[1])


def _reproject_grid(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    src_crs: CRS,
    dst_crs: CRS,
) -> tuple[np.ndarray, np.ndarray]:
    return _reproject(x_coords, y_coords, src_crs, dst_crs)


def _reproject(
    x_coords: NDArray[np.floating] | list[float] | tuple[float, ...],
    y_coords: NDArray[np.floating] | list[float] | tuple[float, ...],
    src_crs: CRS | str,
    dst_crs: CRS | str,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[list[float], list[float]]
    | tuple[tuple[float, ...], tuple[float, ...]]
):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x_coords, y_coords)


def _get_spatial_dims(grid: xr.Dataset | xr.DataArray) -> tuple[str, ...]:
    flat_dim = _get_flat_dim(grid)
    if flat_dim is not None:
        return (flat_dim,)

    spatial_dims = _get_spatial(grid, tuple(grid.dims))  # type: ignore[arg-type]

    if None in spatial_dims:
        raise ValueError(
            f"Spatial dimensions must either be one of {YX}, have CF conventions"
            f" attributes for arbitrary names or be one of {FLAT_DIM} in case of"
            " unstructured grid"
        )

    return spatial_dims  # type: ignore[return-value]


def _set_grid_mapping(ds: xr.Dataset) -> xr.Dataset:
    def serializable_mapping() -> dict[str, Any]:
        return {
            key: float(value) if isinstance(value, np.floating) else value
            for key, value in ds_out[mapname].attrs.items()
        }

    ds_out = ds.copy(deep=False)
    grid_map = set()
    for varname, da in ds_out.items():
        if "grid_mapping" in da.attrs:
            grid_map.add(da.attrs.pop("grid_mapping"))
        elif "grid_mapping_name" in da.attrs:
            grid_map.add(varname)

    if len(grid_map) == 0:
        return ds_out
    if len(grid_map) > 1:
        raise ValueError("Several grid mappings in the same dataset is not supported")

    mapname = grid_map.pop()
    if {"grid_mapping", "crs"}.isdisjoint(ds_out.attrs) and mapname in ds_out:
        ds_out.attrs["grid_mapping"] = json.dumps(serializable_mapping())

    return ds_out.drop_vars(mapname, errors="ignore")


def _get_flat_dim(grid: xr.Dataset | xr.DataArray) -> str | None:
    for dim in FLAT_DIM:
        if dim in grid.dims:
            return dim

    return None


def _combine_attrs(src_attrs: dict, dst_attrs: dict) -> dict:
    from_dst = ("grid_mapping", "crs", "bounding_box")

    combined_attrs = {key: src_attrs[key] for key in SRC_ATTRS_KEEP if key in src_attrs}
    combined_attrs.update({key: dst_attrs[key] for key in from_dst if key in dst_attrs})
    return combined_attrs


def _get_coords_name_crs(
    grid: xr.Dataset | xr.DataArray,
    spatial_dims: tuple[str, ...] | None = None,
) -> tuple[str, str, CRS]:
    latlon = _get_latlon_coords(grid)
    if latlon is not None:
        grid_y, grid_x = latlon
        return grid_y, grid_x, CRS.from_string("EPSG:4326")

    if spatial_dims is None:
        spatial_dims = _get_spatial_dims(grid)

    if len(spatial_dims) == 1:
        grid_y, grid_x = _get_spatial_coords(grid)
        return grid_y, grid_x, _get_grid_crs(grid)

    grid_y, grid_x = spatial_dims
    if grid_y in grid.coords and grid_x in grid.coords:
        return grid_y, grid_x, _get_grid_crs(grid)

    raise ValueError("Missing spatial coordinates")


def _get_grid_crs(grid: xr.Dataset | xr.DataArray) -> CRS:
    if "crs" in grid.attrs:
        return CRS.from_user_input(grid.crs)
    if "grid_mapping" in grid.attrs:
        return _parse_grid_mapping(grid.grid_mapping)
    if _get_latlon_dims(grid) is not None:
        return CRS.from_string("EPSG:4326")

    raise ValueError(
        "The dataset must have a grid_mapping or crs attribute or have latitude"
        " and longitude dimensions"
    )


def _parse_grid_mapping(grid_mapping: str | dict[str, Any]) -> CRS:
    if isinstance(grid_mapping, str):
        grid_mapping = json.loads(grid_mapping)
    if "epsg_code" in grid_mapping:
        return CRS.from_string(grid_mapping["epsg_code"])  # type: ignore[index]

    return CRS.from_cf(grid_mapping)  # type: ignore[arg-type]


def _get_spatial(
    grid: xr.Dataset | xr.DataArray, dim_or_coord: tuple[str, ...]
) -> tuple[str | None, str | None]:
    for y, x in YX:
        if y in dim_or_coord and x in dim_or_coord:
            return y, x

    # look for coordinates with CF attributes in coords
    spatial: list[str | None] = [None, None]
    for name in dim_or_coord:
        attrs = grid[name].attrs
        if (attrs.get("axis", "").upper() == "Y") or (
            attrs.get("standard_name", "").lower()
            in ("latitude", "grid_latitude", "projection_y_coordinate")
        ):
            spatial[0] = name
        elif (attrs.get("axis", "").upper() == "X") or (
            attrs.get("standard_name", "").lower()
            in ("longitude", "grid_longitude", "projection_x_coordinate")
        ):
            spatial[1] = name

        if None not in spatial:
            break

    return (spatial[0], spatial[1])


def _get_spatial_coords(grid: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    notdim_coords = tuple(set(grid.coords) - set(grid.dims))
    spatial_coords = _get_spatial(grid, notdim_coords)  # type: ignore[arg-type]

    if None in spatial_coords:
        raise ValueError(
            "Could not find non-dimension spatial coordinates. Must be one of "
            f"{YX} or have CF conventions attributes for arbitrary names"
        )

    return spatial_coords  # type: ignore[return-value]


def _get_latlon_coords(
    grid: xr.Dataset | xr.DataArray, nondim: bool = True
) -> tuple[str, str] | None:
    coords = tuple(grid.coords)
    if nondim:
        coords = tuple(set(coords) - set(grid.dims))
    return _get_latlon(grid, coords)  # type: ignore[arg-type]


def _get_latlon(
    grid: xr.Dataset | xr.DataArray,
    dim_or_coord: tuple[str, ...],
) -> tuple[str, str] | None:
    for lat, lon in LATLON:
        if lat in dim_or_coord and lon in dim_or_coord:
            return lat, lon
    latlon: list[str | None] = [None, None]
    for name in dim_or_coord:
        coord_attrs = grid[name].attrs
        if (
            coord_attrs.get("standard_name", "") == "latitude"
            or coord_attrs.get("long_name", "") == "latitude"
        ):
            latlon[0] = name
        elif (
            coord_attrs.get("standard_name", "") == "longitude"
            or coord_attrs.get("long_name", "") == "longitude"
        ):
            latlon[1] = name

        if None not in latlon:
            break

    if None in latlon:
        return None

    return (latlon[0], latlon[1])


def _get_latlon_dims(grid: xr.Dataset | xr.DataArray) -> tuple[str, str] | None:
    return _get_latlon(grid, tuple(grid.dims))


# -----------------------------------------------------------------------------
# Dataset / DataArray conversion helpers
# -----------------------------------------------------------------------------
def _ds_to_da(
    ds: xr.Dataset,
) -> tuple[xr.DataArray, dict[Hashable, dict[Hashable, Any]]]:
    vars_attrs = {varname: da.attrs for varname, da in ds.data_vars.items()}
    da = ds.to_array()
    return da, vars_attrs


def _da_to_ds(
    da: xr.DataArray, vars_attrs: dict[Hashable, dict[Hashable, Any]]
) -> xr.Dataset:
    ds = da.to_dataset("variable")
    _update_vars_attrs(ds, vars_attrs)

    return ds


def _update_vars_attrs(
    ds: xr.Dataset, vars_attrs: dict[Hashable, dict[Hashable, Any]]
) -> None:
    for varname, attributes in vars_attrs.items():
        ds[varname].attrs.update(attributes)


# -----------------------------------------------------------------------------
# Subsetting, coarsening, and resolution utilities
# -----------------------------------------------------------------------------
def _slice_coarsen(
    data: xr.Dataset,
    dst_grid: xr.Dataset,
    method: str,
    spatial_dims: Optional[tuple[str, ...]] = None,
    dst_spatial_dims: Optional[tuple[str, ...]] = None,
    **kwargs: Any,
) -> xr.Dataset:
    def extra_buffer(ratio: int) -> int:
        if buffer == 1:
            return ceil(ratio / 2)

        return (ratio - 1) * (buffer - 1)

    if spatial_dims is None:
        spatial_dims = _get_spatial_dims(data)
    if dst_spatial_dims is None:
        dst_spatial_dims = _get_spatial_dims(dst_grid)

    coords_crs = _get_coords_name_crs(data, spatial_dims)
    dst_coords_crs = _get_coords_name_crs(dst_grid, dst_spatial_dims)
    grid_crs, dst_grid_crs = coords_crs[2], dst_coords_crs[2]

    x_coords, y_coords = (
        dst_grid[dst_coords_crs[1]].values,
        dst_grid[dst_coords_crs[0]].values,
    )

    if grid_crs != dst_grid_crs:
        if x_coords.ndim == 1:
            x_coords, y_coords = np.meshgrid(x_coords, y_coords)

        # only keep the outer rectangle
        x_coords = np.append(
            x_coords[[0, -1], :].ravel(), x_coords[1:-1, [0, -1]].ravel()
        )
        y_coords = np.append(
            y_coords[[0, -1], :].ravel(), y_coords[1:-1, [0, -1]].ravel()
        )

        # Reproject destination coordinates in the grid CRS
        x_coords, y_coords = _reproject(x_coords, y_coords, dst_grid_crs, grid_crs)

    yx_bnds = ((y_coords.min(), y_coords.max()), (x_coords.min(), x_coords.max()))

    buffer = _get_buffer(method, **kwargs)
    sub_domain, res = _get_sub_domain(
        data, spatial_dims, coords_crs[:2], yx_bnds, buffer
    )
    data_sub = data.isel(sub_domain)

    # coarsen input dataset if needed (downsampling)
    res_ratio_x, res_ratio_y = _get_coarsen_ratio(
        data_sub,
        spatial_dims,
        coords_crs,
        dst_grid,
        dst_spatial_dims,  # type: ignore[arg-type]
        dst_coords_crs,
        res,
    )
    if (res_ratio_x > 1) or (res_ratio_y > 1):
        if len(spatial_dims) == 1:
            logger.warning(
                "Unstructured grid cannot be coarsened. Interpolation to a lower"
                " resolution grid may create aliasing artifacts."
            )
            return data_sub

        coarsen_specs = {
            dim: ratio
            for dim, ratio in (
                (spatial_dims[1], res_ratio_x),
                (spatial_dims[0], res_ratio_y),
            )
            if ratio > 1
        }
        sub_domain.update(
            {
                dim: slice(  # type: ignore[misc]
                    max(sub_domain[dim].start - extra_buffer(ratio), 0),  # type: ignore[union-attr]
                    sub_domain[dim].stop + extra_buffer(ratio),  # type: ignore[union-attr]
                )
                for dim, ratio in coarsen_specs.items()
            }
        )
        return (
            data.isel(sub_domain)  # type: ignore[attr-defined]
            .coarsen(coarsen_specs, boundary="trim")
            .mean(skipna=False)
        )

    return data_sub


def _get_buffer(method: str, **kwargs: Any) -> int:
    buffer = 1
    if (method == "cubic") or ("3D" in method):
        buffer += 1
    if "k" in kwargs:
        buffer += ceil(np.sqrt(kwargs["k"] / np.pi))
    if "r_search" in kwargs:
        buffer += ceil(kwargs["r_search"])
    return buffer


def _get_sub_domain(
    data: xr.Dataset,
    spatial_dims: tuple[str, ...],
    spatial_coords: tuple[str, str],
    yx_bnds: tuple[tuple[float, float], tuple[float, float]],
    buffer: int,
) -> tuple[dict[str, slice], None] | tuple[dict[str, NDArray], float]:
    def get_cond_xy(buffer_crs: float) -> NDArray[np.bool_]:
        cond_x = (data[grid_x].values > x_bnds[0] - buffer_crs) & (
            data[grid_x].values < x_bnds[1] + buffer_crs
        )
        cond_y = (data[grid_y].values > y_bnds[0] - buffer_crs) & (
            data[grid_y].values < y_bnds[1] + buffer_crs
        )
        return cond_x & cond_y

    def get_ind_startend(
        coordname: str, dim: str, bnds: tuple[float, float]
    ) -> tuple[int, int]:
        coords = data[coordname]
        dim_size = data.sizes[dim]

        bool_decrease = is_decreasing(coords, dim)

        if bool_decrease:
            coords = coords.isel({dim: slice(None, None, -1)})

        # Find start index: first value greater than lower bound
        id_start = (coords > bnds[0]).argmax(dim).values.min().item()  # type: ignore[union-attr]

        # Find end index: first value greater than upper bound, or use full size
        end_mask = coords > bnds[1]
        id_end = (
            end_mask.argmax(dim).values.max().item()  # type: ignore[union-attr]
            if end_mask.values.any()
            else dim_size
        )

        if bool_decrease:
            start_ori = id_start
            id_start = dim_size - id_end
            id_end = dim_size - start_ori

        return id_start, id_end

    grid_y, grid_x = spatial_coords
    y_bnds, x_bnds = yx_bnds

    if len(spatial_dims) == 1:
        res_start = max(
            abs(np.diff(data[grid_x][:2])[0]),
            abs(np.diff(data[grid_y][:2])[0]),
        )
        res_end = max(
            abs(np.diff(data[grid_x][-2:])[0]),
            abs(np.diff(data[grid_y][-2:])[0]),
        )
        res_approx = max(res_start, res_end)
        cond_xy = get_cond_xy(2 * res_approx)
        res = _get_flat_grid_res(data[grid_x][cond_xy], data[grid_y][cond_xy])
        cond_xy = get_cond_xy(buffer * res * 1.5)
        return {spatial_dims[0]: cond_xy}, res

    y_dim, x_dim = spatial_dims
    idx_start, idx_end = get_ind_startend(grid_x, x_dim, x_bnds)
    idy_start, idy_end = get_ind_startend(grid_y, y_dim, y_bnds)
    idx_start = max(idx_start - buffer, 0)
    idx_end += buffer
    idy_start = max(idy_start - buffer, 0)
    idy_end += buffer
    return {x_dim: slice(idx_start, idx_end), y_dim: slice(idy_start, idy_end)}, None


def _get_flat_grid_res(coords_x: xr.DataArray, coords_y: xr.DataArray) -> float:
    def dcoords(coords: xr.DataArray) -> float:
        return np.median(abs(np.diff(coords.values)))

    return max(dcoords(coords_x), dcoords(coords_y))


def _get_coarsen_ratio(
    data: xr.Dataset,
    spatial_dims: tuple[str, ...],
    coords_crs: tuple[str, str, CRS],
    dst_grid: xr.Dataset,
    dst_spatial_dims: tuple[str, str],
    dst_coords_crs: tuple[str, str, CRS],
    res: Optional[float] = None,
) -> tuple[int, int]:
    if res is not None:
        res_src_x = res_src_y = res
    else:
        center_src_x, center_src_y = _center_coords(
            data,
            spatial_dims,
            coords_crs[:2],  # type: ignore[arg-type]
        )
        res_src_x = center_src_x.diff(spatial_dims[1]).values.mean()
        res_src_y = center_src_y.diff(spatial_dims[0]).values.mean()

    center_dst_x, center_dst_y = _center_coords(
        dst_grid, dst_spatial_dims, dst_coords_crs[:2]
    )
    src_crs, dst_crs = coords_crs[2], dst_coords_crs[2]
    if src_crs != dst_crs:
        center_dst_x.values, center_dst_y.values = _reproject(  # type: ignore[assignment]
            center_dst_x.values,
            center_dst_y.values,
            dst_crs,
            src_crs,
        )

    res_dst_x = center_dst_x.diff(dst_spatial_dims[1]).values.mean()
    res_dst_y = center_dst_y.diff(dst_spatial_dims[0]).values.mean()
    return abs(round(res_dst_x / res_src_x)), abs(round(res_dst_y / res_src_y))


def _center_coords(
    grid: xr.Dataset,
    dims: tuple[str, str],
    coords: tuple[str, str],
) -> tuple[xr.DataArray, xr.DataArray]:
    y_dim, x_dim = dims
    grid_y, grid_x = coords
    mid_x = int(grid.sizes[x_dim] / 2)
    mid_y = int(grid.sizes[y_dim] / 2)
    sub_x = grid[grid_x].isel({x_dim: slice(mid_x - 1, mid_x + 1)})
    sub_y = grid[grid_y].isel({y_dim: slice(mid_y - 1, mid_y + 1)})
    if y_dim in sub_x.dims:
        sub_x = sub_x.isel({y_dim: slice(mid_y - 1, mid_y + 1)})
        sub_y = sub_y.isel({x_dim: slice(mid_x - 1, mid_x + 1)})
    else:
        sub_y, sub_x = xr.broadcast(sub_y, sub_x)

    return sub_x, sub_y


# -----------------------------------------------------------------------------
# Flattening and CRS-flatten helpers
# -----------------------------------------------------------------------------
def _flatten_coords_crs(
    grid: xr.DataArray | xr.Dataset,
    spatial_dims: tuple[str, ...],
    flat_dim: str = "cell",
) -> tuple[xr.DataArray | xr.Dataset, str, str, CRS]:
    if len(spatial_dims) > 1:
        # flatten grid
        grid = grid.stack({flat_dim: spatial_dims}).transpose(flat_dim, ...)
        spatial_dims = (flat_dim,)

    grid_y, grid_x, grid_crs = get_coords_name_crs(grid, spatial_dims)

    return grid, grid_y, grid_x, grid_crs


def get_surf_alt(grid: xr.Dataset | xr.DataArray) -> str:
    for name in SURF_ALT:
        if name in grid.coords:
            return name

    raise ValueError(f"Missing one of {SURF_ALT} coordinate variable")


@memoize(6)
def _reproject_grid(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    src_crs: CRS,
    dst_crs: CRS,
) -> tuple[np.ndarray, np.ndarray]:
    return _reproject(x_coords, y_coords, src_crs, dst_crs)  # type: ignore[return-value]


def _reproject_grid_cached(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    src_crs: CRS,
    dst_crs: CRS,
) -> tuple[np.ndarray, np.ndarray]:
    if dst_crs != src_crs:
        hkey = (x_coords.tobytes(), y_coords.tobytes(), x_coords.shape, dst_crs)
        x_coords, y_coords = _reproject_grid(
            x_coords, y_coords, src_crs, dst_crs, hkey=hkey
        )

    return x_coords, y_coords


def _flatten_concat_grids(
    data: xr.DataArray,
    spatial_dims: tuple[str, ...],
    dst_grid: xr.Dataset,
    dst_spatial_dims: tuple[str, str],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _choose_crs_meters() -> CRS:
        if reproj_dst:
            if grid_crs in SWISS_CRS:
                return grid_crs
            return (
                CRS.from_string(SWISS_CRS[1])
                if dst_grid_crs == SWISS_CRS[1]
                else CRS.from_string(SWISS_CRS[0])
            )

        if dst_grid_crs in SWISS_CRS:
            return dst_grid_crs
        return (
            CRS.from_string(SWISS_CRS[1])
            if grid_crs == SWISS_CRS[1]
            else CRS.from_string(SWISS_CRS[0])
        )

    data, grid_y, grid_x, grid_crs = _flatten_coords_crs(data, spatial_dims)
    flat_dim = spatial_dims[0] if len(spatial_dims) == 1 else "cell"
    dst_grid, dst_grid_y, dst_grid_x, dst_grid_crs = _flatten_coords_crs(
        dst_grid, dst_spatial_dims, flat_dim
    )

    reproj_dst = dst_grid.sizes[flat_dim] <= data.sizes[flat_dim]

    if method in HEIGHT_METHODS:
        surf_alt = get_surf_alt(data)
        dst_surf_alt = get_surf_alt(dst_grid)
        # include elevation
        coords_flat = np.c_[
            data[grid_x].values,
            data[grid_y].values,
            data[surf_alt].values,
        ]
        dst_coords_flat = np.c_[
            dst_grid[dst_grid_x].values,
            dst_grid[dst_grid_y].values,
            dst_grid[dst_surf_alt].values,
        ]
        target_crs = _choose_crs_meters()
        coords_flat[:, 0], coords_flat[:, 1] = _reproject_grid_cached(
            coords_flat[:, 0],
            coords_flat[:, 1],
            grid_crs,
            target_crs,
        )
        dst_coords_flat[:, 0], dst_coords_flat[:, 1] = _reproject_grid_cached(
            dst_coords_flat[:, 0],
            dst_coords_flat[:, 1],
            dst_grid_crs,
            target_crs,
        )
        return coords_flat, data.values, dst_coords_flat

    # Non-height methods
    coords_flat = np.c_[data[grid_x].values, data[grid_y].values]
    dst_coords_flat = np.c_[dst_grid[dst_grid_x].values, dst_grid[dst_grid_y].values]
    if reproj_dst:  # reproject destination coordinates in the grid CRS
        dst_coords_flat[:, 0], dst_coords_flat[:, 1] = _reproject_grid_cached(
            dst_coords_flat[:, 0],
            dst_coords_flat[:, 1],
            dst_grid_crs,
            grid_crs,
        )
        return coords_flat, data.values, dst_coords_flat

    # reproject grid coordinates in the destination CRS
    coords_flat[:, 0], coords_flat[:, 1] = _reproject_grid_cached(
        coords_flat[:, 0],
        coords_flat[:, 1],
        grid_crs,
        dst_grid_crs,
    )
    return coords_flat, data.values, dst_coords_flat


# -----------------------------------------------------------------------------
# Nearest-neighbour search and interpolation dispatch table
# -----------------------------------------------------------------------------
def _nn_lookup(
    grid_coords: np.ndarray, points: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    tree = KDTree(grid_coords)
    dd, ii = tree.query(points, k, workers=-1)

    return dd.astype(points.dtype, copy=False), ii


INTERP_FUNC = {
    "nearest_alt": _interp_nearest_alt,
    "nearest": _interp_nearest,
    "nearest_3D": _interp_nearest,
    "idw": _interp_idw,
}
