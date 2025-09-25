===========================
anemoi-weathermart-plugins
===========================
Collection of [anemoi plugins](https://anemoi.readthedocs.io/projects/plugins/en/latest/index.html#) for diverse data sources.

This package bridges:

    - sources accessible from the **weathermart** package to **anemoi.sources**. As of now, sources that you can write in a yaml file are the following:

        - numerical weather prediction analyses or forecasts from *any grib archive*;
        - topography: ``nasadem``, ``cedtm`` and ``dhm25`` using url requests from original sources;
        - observations: ``station`` data from **jretrieve api**;
        - satellite: ``satellite`` MSG variables from EUMETSAT API;
        - radar: ``radar`` data from OPERA API.

In the above description, text ``formatted like this`` are sources that can be written in a yaml configuration file for creating a dataset using **anemoi-datasets**.

The objective of this package is to use the **anemoi-datasets** package with *custom sources*.