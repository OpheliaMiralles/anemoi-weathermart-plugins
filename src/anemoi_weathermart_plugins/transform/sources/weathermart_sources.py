import datetime
import logging
from itertools import chain
from typing import Any

import pandas as pd
import xarray as xr
from anemoi.datasets.create.source import Source
from anemoi.datasets.dates.groups import GroupOfDates
from weathermart import DataProvider
from weathermart.default_provider import available_retrievers
from weathermart.default_provider import default_provider

from anemoi_weathermart_plugins.helpers import assign_lonlat
from anemoi_weathermart_plugins.xarray_extensions import CustomFieldList
from anemoi_weathermart_plugins.xarray_extensions import check_indexing


def get_all_available_sources() -> list[str]:
    return list(set(chain.from_iterable(r.sources for r in available_retrievers())))


def get_fieldlist_from_data_provider(
    provider: DataProvider,
    source: str,
    dates: list[datetime.datetime],
    param: list[str] | None = None,
    **retriever_kwargs: Any,
) -> CustomFieldList:
    expanded_kwargs = retriever_kwargs.copy()
    resample = expanded_kwargs.pop("resample", None)
    rename = expanded_kwargs.pop("rename", None)
    data = provider.provide(source, param, dates, **expanded_kwargs).drop_duplicates(
        ...
    )
    time_dim = (
        "forecast_reference_time"
        if "forecast_reference_time" in data.dims
        else "time"
        if "time" in data.dims
        else None
    )
    if time_dim is None:
        raise ValueError(
            "No time dimension found in the provided data. Must be one of 'time' or 'forecast_reference_time'."
        )
    data = check_indexing(data, time_dim)
    time_dim = "time"  # after check_indexing, time_dim is always "time"
    if resample is not None:
        logging.info("Resampling data to %s frequency.", resample)
        data_next = provider.provide(
            source, param, [d + pd.to_timedelta("1d") for d in dates], **expanded_kwargs
        ).drop_duplicates(
            ...
        )  # otherwise we don't get reampled data at the end of the period
        data_next = check_indexing(
            data_next,
            time_dim="forecast_reference_time"
            if "forecast_reference_time" in data_next.dims
            else "time",
        )
        data = xr.concat([data, data_next], dim=time_dim)
        data = data.resample({time_dim: resample}).interpolate("linear")
    crs = provider.get_crs(source)
    if isinstance(crs, dict) and "through" in expanded_kwargs:
        crs = crs[expanded_kwargs["through"]]
    if time_dim is not None:  # e.g. not a forcing dataset
        data = data.sortby(time_dim).sel(
            {time_dim: dates}, method="nearest"
        )  # return only the relevant time for daily arrays
        data[time_dim] = dates
    else:
        data = data.assign_coords(
            time=dates
        )  # if no time dimension, assign the dates as a coordinate
        time_dim = "time"
    if not ("longitude" in data.coords and "latitude" in data.coords):
        data = assign_lonlat(data, crs)
    if (
        len(dates) == 1
    ):  # minimal input selects first day at midnight even if time is missing...
        first_hour_day = [d for d in data[time_dim].dt.round("1d").to_numpy()]
        data = data.reindex_like(
            xr.Dataset(coords={time_dim: first_hour_day}), method="nearest"
        )
    if rename is not None:
        data = data.rename(rename)
    xarray_fieldlist = CustomFieldList.from_xarray(data, proj_string=crs, source=source)
    return xarray_fieldlist


class DataProviderSource(Source):
    """Base source class for data provider sources in Anemoi."""

    def __init__(self, context, source, param, **retriever_kwargs):
        super().__init__(
            context, source, param=param, retriever_kwargs=retriever_kwargs
        )
        self.provider = default_provider(cache_location="/store_new/mch/msclim/pronos")
        self.source = source
        self.param = param
        self.retriever_kwargs = retriever_kwargs

    def execute(self, dates: list[datetime.datetime]):
        if isinstance(dates, GroupOfDates):
            dates = dates.dates
        if not isinstance(self.param, list):
            self.param = [self.param]
        fieldlist = get_fieldlist_from_data_provider(
            self.provider,
            self.source,
            dates,
            self.param,
            **self.retriever_kwargs,
        )
        return fieldlist


def make_source_class(source_name: str):
    def __init__(self, context, param, **retriever_kwargs):
        super(self.__class__, self).__init__(
            context, source_name, param, **retriever_kwargs
        )

    class_name = source_name.replace("-", "_")
    return type(class_name, (DataProviderSource,), {"__init__": __init__})


def get_all_source_classes(source_names: list[str]) -> dict[str, type]:
    return {name: make_source_class(name) for name in source_names}


source_names = get_all_available_sources()
source_classes = get_all_source_classes(source_names)
# This will print the source class definitions
# Still have to execute this code to make sure the classes are created and available for the pyproject
print(
    "\n".join(
        f'{name.replace("-", "_")} = make_source_class("{name}")'
        for name in source_names
    )
)

DHM25 = make_source_class("DHM25")
OPERA = make_source_class("OPERA")
NASADEM = make_source_class("NASADEM")
SATELLITE = make_source_class("SATELLITE")
SURFACE = make_source_class("SURFACE")
INCA = make_source_class("INCA")
ICON_CH1 = make_source_class("ICON-CH1-EPS")
