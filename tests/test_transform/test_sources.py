import entrypoints
import pytest
from anemoi.datasets.create import Init
from data_provider.default_provider import all_retrievers

from anemoi_weathermart_plugins.transform.sources import DataProviderSource

all_entrypoints = entrypoints.get_group_all("anemoi.datasets.sources")
all_sources = [e.name for e in all_entrypoints]
all_classes = [e.object_name.replace("_", "-") for e in all_entrypoints]
example_variable = {
    e: next(list(r.variables.keys())[0] for r in all_retrievers() if s in r.sources)
    for (s, e) in zip(all_classes, all_sources)
}
example_variable["satellite"] = (
    "IR_039"  # satellite data exist for GridefixRetriever so the  ifirst variable is air_temperature
)
source_kwargs = {
    "satellite": {"through": "eumetsat"},
}


@pytest.mark.parametrize("source_name", all_sources)
def test_data_provider_sources(source_name, tmp_path):
    """Test that all data-provider sources can be loaded."""
    source = entrypoints.get_single("anemoi.datasets.sources", source_name)
    assert source is not None, f"Source {source_name} could not be loaded"

    # test if can be instantiated
    try:
        instance = source.load()
        assert issubclass(instance, DataProviderSource), (
            f"Instance of {source_name} could not be created"
        )
    except Exception as e:
        pytest.fail(f"Failed to create instance of {source_name}: {e}")

    # test if we can create a dataset with this source
    dataset_test_config = {
        "dates": {"start": "2024-09-02", "end": "2024-09-03", "frequency": "1h"},
        "input": {source.name: {"param": [example_variable[source.name]]}},
        "build": {"group_by": "daily", "variable_naming": "param", "allow_nans": True},
    }
    if source.name in source_kwargs:
        dataset_test_config["input"][source.name].update(source_kwargs[source.name])
    try:
        c = Init(
            path=str(tmp_path.with_suffix(".zarr")),
            config=dataset_test_config,
            overwrite=True,
        )
        c.run()  # if problems with run the data_provider is the cause
    except Exception as e:
        pytest.fail(f"Failed to run dataset initialization for {source_name}: {e}")
