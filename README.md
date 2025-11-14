# anemoi-weathermart-plugins

Collection of [anemoi plugins](https://anemoi.readthedocs.io/projects/plugins/en/latest/index.html#) for diverse data sources.

This package bridges:

- sources accessible from **[weathermart](https://github.com/OpheliaMiralles/weathermart)** to **anemoi.sources**.  
  As of now, sources that you can write in a YAML file are the following:
  - numerical weather prediction analyses or forecasts from *any GRIB archive*;
  - topography: `nasadem`, `cedtm`, and `dhm25` using URL requests from original sources;
  - satellite: `satellite` MSG variables from the EUMETSAT API;
  - radar: `radar` data from the OPERA API.

It also provides filters such as `interp2grid` and `interp2res` to harmonize data from different sources.

In the above description, text formatted like `this` are source names that can be written in a YAML configuration file for creating a dataset using **anemoi-datasets**.

The objective of this package is to use the **anemoi-datasets** package with *custom sources*.

**Related project:** [weathermart (GitHub)](https://github.com/OpheliaMiralles/weathermart)
