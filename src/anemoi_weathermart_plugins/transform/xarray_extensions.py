import json
from typing import Any

import numpy as np
import xarray as xr
import yaml
from anemoi.datasets.create.sources.xarray_support.field import XArrayField
from anemoi.datasets.create.sources.xarray_support.fieldlist import XarrayFieldList
from anemoi.datasets.create.sources.xarray_support.flavour import CoordinateGuesser
from anemoi.datasets.create.sources.xarray_support.time import Time
from anemoi.datasets.create.sources.xarray_support.variable import Variable
from pyproj import CRS

from anemoi_weathermart_plugins.helpers import reproject


class CustomVariable(Variable):
    """A variable class for MCH data, extending the XArrayVariable class with more metadata."""

    def __init__(
        self,
        *,
        ds: xr.Dataset,
        var: xr.DataArray,
        coordinates: list[Any],
        grid: xr.Dataset,
        time: Time,
        metadata: dict[Any, Any],
        proj_string: str | None = None,
        source: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Initialize CustomVariable.

        Args:
            ds (xr.Dataset): Input dataset.
            var (xr.DataArray): Variable data array.
            coordinates (list[Any]): List of coordinates.
            grid (xr.Dataset): Grid dataset.
            time (Time): Time object.
            metadata (dict[Any, Any]): Metadata dictionary.
            proj_string (Union[str, None], optional): Projection string. Default is None.
            source (str, optional): Source string. Default is "".
            **kwargs (Any): Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(
            ds=ds,
            variable=var,
            coordinates=coordinates,
            grid=grid,
            time=time,
            metadata=metadata,
            **kwargs,
        )
        self.proj_string = proj_string
        self.source = source
        self._metadata = {
            k.replace("variable", "param"): v for k, v in self._metadata.items()
        }

    def __getitem__(self, i: int) -> "CustomField":
        if i >= self.length:
            raise IndexError(i)
        coords = np.unravel_index(i, self.shape)
        kwargs = {k: v for k, v in zip(self.names, coords)}
        return CustomField(self, self.variable.isel(kwargs))


class CustomField(XArrayField):
    @property
    def source(self) -> str:
        return self.owner.source

    @property
    def dims(self) -> list[str]:
        return list(self.selection.dims)

    @property
    def proj_string(self) -> str:
        return self.owner.proj_string

    @property
    def grid_coords(self) -> np.ndarray:
        std_grid = ["x", "y", "longitude", "latitude"]
        aux = ["cell", "station"]
        return np.intersect1d(
            std_grid + aux, list(self.selection.coords) + list(self.selection.dims)
        )

    @property
    def not_grid_dim(self) -> list[str]:
        return [d for d in self.selection.dims if d not in self.grid_coords]

    @property
    def resolution(self) -> str:
        """
        Compute the resolution based on the minimal spacing along grid dimensions.

        For projected CRS, it computes the minimal difference in x and y (converted to meters)
        and rounds it to a kilometer value.
        """
        spatial_dim = (
            ["x", "y"]
            if "x" in self.selection.dims and "y" in self.selection.dims
            else ["longitude", "latitude"]
        )
        x, y = spatial_dim
        valid_crs = CRS.from_user_input(self.selection.attrs.get("crs", self.crs))
        valid_crs = CRS.from_user_input(
            "epsg:2056"
        )  # force Swiss projection for meters
        minimal = self.selection.isel({d: 0 for d in self.not_grid_dim}).rio.write_crs(
            valid_crs
        )
        if minimal.rio.crs.is_projected:
            minx = np.diff(np.sort(minimal[x].to_numpy())).min()
            miny = np.diff(np.sort(minimal[y].to_numpy())).min()
            res_m = np.array([minx, miny]) * minimal.rio.crs.units_factor[1]
        else:
            res_deg = np.array(minimal.rio.resolution())
            scale = np.diff(
                reproject([0, 1], [0, 0], valid_crs, CRS.from_user_input("epsg:2056"))
            )[0][0]
            res_m = res_deg * scale
        res_km = np.round(res_m / 1e3, 0)
        return f"{res_km[0].item()}km"

    @property
    def crs(self) -> str:
        return self.proj_string

    @property
    def bounding_box(self) -> tuple:
        # (min_x, min_y, max_x, max_y)
        spatial_dim = (
            ["x", "y"]
            if {"x", "y"}.issubset(self.selection.dims)
            else ["longitude", "latitude"]
        )
        x, y = spatial_dim
        minimal = self.selection.isel({d: 0 for d in self.not_grid_dim})
        return (
            float(np.min(minimal[x])),
            float(np.min(minimal[y])),
            float(np.max(minimal[x])),
            float(np.max(minimal[y])),
        )


class CustomFieldList(XarrayFieldList):
    @classmethod
    def from_xarray(
        cls,
        ds: xr.Dataset,
        flavour: str | dict | None = None,
        proj_string: str | None = None,
        source: str = "",
    ) -> "CustomFieldList":
        variables: list[CustomVariable] = []
        if isinstance(flavour, str):
            with open(flavour) as f:
                flavour = (
                    yaml.safe_load(f)
                    if flavour.endswith((".yaml", ".yml"))
                    else json.load(f)
                )
        guess = CoordinateGuesser.from_flavour(ds, flavour)
        skip: set[str] = set()

        def _skip_attr(v: Any, attr_name: str) -> None:
            attr_val = getattr(v, attr_name, "")
            if isinstance(attr_val, str):
                skip.update(attr_val.split(" "))

        for name in ds.data_vars:
            v = ds[name]
            _skip_attr(v, "coordinates")
            _skip_attr(v, "bounds")
            _skip_attr(v, "grid_mapping")
        for name in ds.data_vars:
            if name in skip:
                continue
            v = ds[name]
            v.attrs.update(crs=proj_string)
            coordinates = []
            for coord in v.coords:
                c = guess.guess(ds[coord], coord)
                assert c, f"Could not guess coordinate for {coord}"
                if coord not in v.dims:
                    c.is_dim = False
                coordinates.append(c)
            variables.append(
                CustomVariable(
                    ds=ds,
                    var=v,
                    coordinates=coordinates,
                    grid=guess.grid(coordinates, variable=v),
                    time=Time.from_coordinates(coordinates),
                    metadata={},
                    proj_string=proj_string,
                    source=source,
                )
            )
        return cls(ds, variables)


# ===== Indexing helpers =====
def check_indexing(data: xr.Dataset, time_dim: str) -> xr.Dataset:
    """Helpers function to check and remove unsupported coordinates and dimensions.
    In particular, in the case of analysis data from gridefix, 1 lead time of 0 is provided as a coord, while the forecast_reference_time is the time dimension. However, anemoi does not support lead_time as a coordinate if only one value is provided (the ndarray of lead times has dim 0). We remove lead_time and set the time dimension to forecast_reference_time.
    Otherwise, observation data is indexed by time, and forecast data by unique forecast_reference_time+lead_time, which anemoi supports. The time dimension is renamed to time for simplicity."""
    potentially_misleading_coords = [
        "surface_altitude",
        "land_area_fraction",
        "stationName",
        "dataOwner",
    ]  # those will make anemoi-datasets display warnings for unsupported coordinates
    for n in potentially_misleading_coords:
        if n in data.coords and n not in data.dims:
            data = data.drop(n)
    if "number" not in data.dims:
        if "realization" in data.dims:
            # realization dimension is used for ensemble members
            # but anemoi expects a number dimension
            data = data.rename({"realization": "number"})
        else:
            data = data.expand_dims(number=[0])
    if time_dim == "forecast_reference_time":
        if "lead_time" in data.dims:
            # forecast data: not supported by anemoi at the moment so must
            # check if each valid_time is unique
            valid = (data[time_dim] + data["lead_time"]).values
            if valid.size != np.unique(valid).shape[0]:
                raise ValueError(
                    "Forecast data with lead_time dimension is not supported if valid_time values are not unique."
                )
            # rename forecast_reference_time to time for anemoi compatibility
            # ensure that lead_time dimension is removed for proper selection below
            data = (
                data.assign_coords(time=((time_dim, "lead_time"), valid))
                .stack(z=(time_dim, "lead_time"))
                .swap_dims(z="time")
                .drop_vars(["z", "lead_time", time_dim])
            )
            time_dim = "time"
        else:
            # wrongly indexed observation/analysis data
            if "lead_time" in data.coords:
                data = data.drop_vars("lead_time")  # misleading for anemoi, will try to
                # interpret it as forecast data
            if "time" in data.coords:
                data = data.drop_vars(
                    "time"
                )  # remove time coordinate otherwise 2 coordinates are intepreted as time and anemoi complains
            data = data.rename(
                {"forecast_reference_time": "time"}
            )  # simpler to combine analysis with observation data
    if time_dim == "time":
        # observation data
        data["time"].attrs.update(standard_name="time")
    return data
