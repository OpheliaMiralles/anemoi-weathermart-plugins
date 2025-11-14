import re
from typing import Union

import numpy as np
import xarray as xr
from anemoi.transform.filter import Filter
from earthkit.data.indexing.fieldlist import FieldArray
from pyproj import CRS
from scipy.interpolate import NearestNDInterpolator

from anemoi_weathermart_plugins.helpers import assign_lonlat
from anemoi_weathermart_plugins.helpers import reproject
from anemoi_weathermart_plugins.transform.filters._vendor.interp2grid import interp2grid
from anemoi_weathermart_plugins.xarray_extensions import CustomFieldList


def merge_fieldlist(field_array: FieldArray) -> xr.Dataset:
    """Merge a FieldArray into a single xarray.Dataset grouped by forecast_reference_time."""
    time_dim = (
        "time"
        if "time" in field_array[0].selection.dims
        else "forecast_reference_time"
        if "forecast_reference_time" in field_array[0].selection.dims
        else None
    )
    if time_dim is None and "time" in field_array[0].selection.coords:
        time_dim = "time"
    times = sorted(set(field.metadata(time_dim) for field in field_array))
    merged = []
    for t in times:
        datasets = [
            field.selection.to_dataset()
            for field in field_array
            if field.metadata(time_dim) == t
        ]
        merged.append(xr.merge(datasets))
    return xr.concat(merged, dim=time_dim)


def _interp2grid(
    array: xr.Dataset, example_field, template: Union[xr.Dataset, str]
) -> xr.Dataset:
    time_dim = (
        "time"
        if "time" in array.dims
        else "forecast_reference_time"
        if "forecast_reference_time" in array.dims
        else None
    )
    point_dim = (
        "cell"
        if "cell" in array.dims
        else "station"
        if "station" in array.dims
        else None
    )
    if point_dim is not None:
        method = "idw"
        kwargs = {"k": 4}
    else:
        method = "linear"
        kwargs = {}
    if isinstance(template, str) and template.startswith("$file:"):
        template = xr.open_zarr(template.removeprefix("$file:"))
    template["x"] = template["x"].astype(np.float32)
    template["y"] = template["y"].astype(np.float32)
    intermediate = array
    if "station" in array.dims:
        intermediate = intermediate.rename({"station": "cell"})
    ds_from_array = intermediate.assign_attrs(
        {"source": example_field.source, "crs": example_field.crs}
    )
    interpolated_array = interp2grid(
        ds_from_array, dst_grid=template, method=method, **kwargs
    ).chunk("auto")
    interpolated_array = interpolated_array.assign_attrs({"crs": template.crs})
    if (
        "longitude" not in interpolated_array.coords
        or "latitude" not in interpolated_array.coords
    ):
        interpolated_array = assign_lonlat(interpolated_array, template.crs)
    for v in interpolated_array.data_vars:
        interpolated_array[v].attrs["crs"] = template.crs
    return interpolated_array.transpose(time_dim, "number", ...)


def _interp_na(array: xr.Dataset, param: str) -> xr.Dataset:
    da_vals = array[param].to_numpy()
    indices = np.where(np.isfinite(da_vals))
    nans = np.isnan(da_vals)
    interp = NearestNDInterpolator(np.transpose(indices), da_vals[indices])
    da_vals[nans] = interp(*np.where(nans))
    array[param].data = da_vals
    return array


def _interp2res(
    array: xr.Dataset, example_field, resolution: Union[str, int], target_crs=None
) -> xr.Dataset:
    point_dim = (
        "cell"
        if "cell" in array.dims
        else "station"
        if "station" in array.dims
        else None
    )
    if point_dim is not None:
        method = "idw"
        kwargs = {"k": point_dim}
    else:
        method = "linear"
    resolution_km = float(re.sub(r"[^0-9.\-]", "", str(resolution)))
    example_crs = example_field.crs
    target_crs = target_crs or example_crs
    _xmin, _ymin, _xmax, _ymax = example_field.bounding_box
    if target_crs != example_crs:
        [_xmin, _xmax], [_ymin, _ymax] = reproject(
            [_xmin, _xmax],
            [_ymin, _ymax],
            CRS.from_user_input(example_crs),
            CRS.from_user_input(target_crs),
        )
    resolution_in_crs_units = np.diff(
        reproject(
            [0, 1000],
            [0, 0],
            CRS.from_user_input("epsg:2056"),
            CRS.from_user_input(target_crs),
        )
    )[0][0]
    template = xr.Dataset(
        coords={
            "x": (
                np.arange(_xmin, _xmax, resolution_in_crs_units, dtype=np.float64)
                + resolution_in_crs_units / 2
            ),
            "y": (
                np.arange(_ymin, _ymax, resolution_in_crs_units, dtype=np.float64)
                + resolution_in_crs_units / 2
            ),
        },
        attrs={"crs": target_crs},
    )
    ds_from_array = array.assign_attrs(
        {"source": example_field.source, "crs": example_crs}
    )
    interpolated_array = interp2grid(
        ds_from_array, dst_grid=template, method=method, **kwargs
    ).chunk("auto")
    interpolated_array = assign_lonlat(interpolated_array, target_crs)
    interpolated_array = interpolated_array.interpolate_na("x")
    interpolated_array.attrs["resolution"] = resolution_km
    interpolated_array.attrs["crs"] = target_crs
    return interpolated_array


class BaseXarrayFilter(Filter):
    api_version = "1.0.0"
    schema = None

    def forward(self, field_array: FieldArray) -> FieldArray:
        """Merge fields, apply the filter, and return a new FieldArray."""
        example = field_array[0]
        merged = (
            field_array.ds
            if hasattr(field_array, "ds")
            else merge_fieldlist(field_array)
        )
        result = self.apply_filter(merged, example)
        return CustomFieldList.from_xarray(
            result, proj_string=example.crs, source=example.source
        )

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        """Override in subclass with the actual transformation."""
        raise NotImplementedError


class Interp2Grid(BaseXarrayFilter):
    """Interpolate fields to a target grid."""

    def __init__(self, template: Union[xr.Dataset, str]):
        self.template = template

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        return _interp2grid(ds, example_field, template=self.template)


class InterpNAFilter(BaseXarrayFilter):
    """Fill NaN values for a given parameter using nearest neighbor interpolation."""

    def __init__(self, param: str):
        self.param = param

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        return _interp_na(ds, self.param)


class Interp2Res(BaseXarrayFilter):
    """Interpolate fields to a target resolution."""

    def __init__(self, resolution: Union[str, int], target_crs: str = None):
        self.resolution = resolution
        self.target_crs = target_crs

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        return _interp2res(
            ds, example_field, resolution=self.resolution, target_crs=self.target_crs
        )
