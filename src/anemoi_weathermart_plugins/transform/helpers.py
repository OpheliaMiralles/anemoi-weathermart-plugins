import io
from copy import copy
from typing import Callable
from typing import Iterator

import earthkit.data as ekd
import numpy as np
import xarray as xr
from meteodatalab import data_source
from meteodatalab import grib_decoder
from pyproj import CRS
from pyproj import Transformer


def reproject(
    x_coords: np.ndarray | list | tuple,
    y_coords: np.ndarray | list | tuple,
    src_crs: CRS | str,
    dst_crs: CRS | str,
):
    # Local copy to avoid circular import with interp2grid/destaggering
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x_coords, y_coords)


def replace(instance, **kwargs):
    new_instance = copy(instance)
    for k, v in kwargs.items():
        if hasattr(new_instance, k):
            setattr(new_instance, k, v)
        else:
            raise AttributeError(f"Attribute '{k}' does not exist in the instance.")
    return new_instance


def assign_lonlat(array: xr.DataArray, crs: str) -> xr.DataArray:
    # Supports (x,y) or (y,x); falls back to reprojection if CRS not WGS84.
    geodims = [d for d in array.dims if d in ("x", "y", "station", "cell")]
    if len(geodims) < 2 and crs == "epsg:4326" and "x" in array.coords:
        return array.assign_coords(longitude=array.x.values, latitude=array.y.values)
    if geodims == ["y", "x"]:
        xv, yv = np.meshgrid(array.x.values, array.y.values)
        lon, lat = (
            (xv, yv)
            if crs == "epsg:4326"
            else reproject(xv, yv, crs, CRS.from_user_input("epsg:4326"))
        )
        return array.assign_coords(
            longitude=(("y", "x"), lon), latitude=(("y", "x"), lat)
        )
    xv, yv = np.meshgrid(array.x.values, array.y.values, indexing="ij")
    lon, lat = (
        reproject(xv, yv, crs, CRS.from_user_input("epsg:4326"))
        if crs != "epsg:4326"
        else (xv, yv)
    )
    return array.assign_coords(longitude=(("x", "y"), lon), latitude=(("x", "y"), lat))


class FieldListDataSource(data_source.DataSource):
    """Adapter exposing an earthkit FieldList as a meteodatalab DataSource."""

    def __init__(self, fieldlist: ekd.FieldList):
        """Initialize the data source.

        Args:
            fieldlist (ekd.FieldList): FieldList to serve via the retrieval API.
        """
        self.fieldlist = fieldlist

    def _retrieve(self, request: dict) -> Iterator[object]:
        """Yield fields matching the request.

        Args:
            request (dict): Selection arguments passed to FieldList.sel(**request).

        Returns:
            Iterator[object]: Iterator over matching fields from the underlying FieldList.
        """
        yield from self.fieldlist.sel(**request)


def meteodatalab_wrapper(
    func: Callable[..., dict[str, xr.DataArray]],
) -> Callable[[ekd.FieldList], ekd.FieldList]:
    """Decorator to wrap a function that processes an ekd.FieldList."""

    def inner(fieldlist: ekd.FieldList) -> ekd.FieldList:
        source = FieldListDataSource(fieldlist)
        result = func(source)
        return _meteodalab_ds_to_fieldlist(result)

    return inner


def to_meteodatalab(fieldlist: ekd.FieldList) -> dict[str, xr.DataArray]:
    """Convert an ekd.FieldList to a dictionary of xarray DataArrays."""
    source = FieldListDataSource(fieldlist)
    return grib_decoder.load(source, {})


def from_meteodatalab(ds: dict[str, xr.DataArray]) -> ekd.FieldList:
    """Convert a dictionary of xarray DataArrays to an ekd.FieldList."""
    return _meteodalab_ds_to_fieldlist(ds)


def _meteodalab_ds_to_fieldlist(ds: dict[str, xr.DataArray]) -> ekd.FieldList:
    with io.BytesIO() as buffer:
        # write data to the buffer
        for da in ds.values():
            if "z" in da.dims and da["z"].size == 1 and bool(da["z"].values is None):
                da = da.squeeze("z", drop=True)
            grib_decoder.save(
                da, buffer, bits_per_value=32
            )  # TODO: find out why we need 32 and 16 leads to precision loss

        # reset the buffer position to the beginning
        buffer.seek(0)

        # read data from the buffer into a FieldList
        fs = ekd.from_source("stream", buffer, read_all=True, lazily=False)

        # somehow read_all does not work correctly, so we need to convert to FieldList
        # to actually have all data loaded in memory and not get IO errors later
        fl = ekd.FieldList.from_fields(fs)

    return fl
