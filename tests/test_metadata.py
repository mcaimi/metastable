import pytest

from metastable.pipelines import GenerationMetadata, format_metadata


def test_attributes_from_metadata():
    meta = GenerationMetadata({
        "output_parameters": format_metadata("cat", "dog", steps=5, seed=3, scheduler="Euler"),
        "loras": {},
    })
    assert meta.prompt == "cat"
    assert meta.negative_prompt == "dog"
    assert meta.num_inference_steps == 5
    assert meta.seed == 3
    assert meta.scheduler == "Euler"
    assert meta.loras == {}
    assert meta.metadata["loras"] == {}


@pytest.mark.parametrize("bad", [
    {},
    {"output_parameters": {}},
    {"output_parameters": {"instances": []}},
    {"output_parameters": None},
])
def test_invalid_metadata(bad):
    with pytest.raises(ValueError, match="Cannot load metadata"):
        GenerationMetadata(bad)
