from types import SimpleNamespace
from typing import ClassVar

import pytest

from kernl.models.discovery import DiscoveryError, HuggingFaceDiscovery


def test_model_mapping():
    class Model:
        id = "org/model"
        author = "org"
        downloads = 12
        likes = 3
        pipeline_tag = "text-generation"
        library_name = "transformers"
        tags: ClassVar[list[str]] = ["text-generation", "gguf"]
        gated = False

    result = HuggingFaceDiscovery._map(Model())
    assert result.repo_id == "org/model"
    assert result.downloads == 12
    assert "gguf" in result.tags


def test_search_rejects_invalid_limit():
    discovery = HuggingFaceDiscovery(api=object())
    try:
        discovery.search(limit=0)
    except ValueError as exc:
        assert "between 1 and 100" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_inspect_collects_sorted_gguf_variants_and_metadata():
    info = SimpleNamespace(
        id="org/model",
        sha="abc123",
        gated=False,
        card_data={"license": "apache-2.0"},
        siblings=[
            SimpleNamespace(rfilename="README.md", size=10, lfs=None),
            SimpleNamespace(rfilename="model.Q4_K_M.gguf", size=4_000_000, lfs=None),
            SimpleNamespace(
                path="sub/model.fp16.gguf", size=None, lfs={"size": 8_000_000}
            ),
            SimpleNamespace(rfilename="model.Q8_0.gguf", size=1_000_000, lfs=None),
        ],
    )
    api = SimpleNamespace(model_info=lambda *args, **kwargs: info)

    result = HuggingFaceDiscovery(api=api).inspect("org/model", revision="main")

    assert result.repo_id == "org/model"
    assert result.revision == "abc123"
    assert result.license == "apache-2.0"
    assert result.gated is False
    assert [variant.filename for variant in result.variants] == [
        "model.Q8_0.gguf",
        "model.Q4_K_M.gguf",
        "sub/model.fp16.gguf",
    ]
    assert [variant.quantization for variant in result.variants] == [
        "Q8_0",
        "Q4_K_M",
        "FP16",
    ]
    assert [variant.size_bytes for variant in result.variants] == [
        1_000_000,
        4_000_000,
        8_000_000,
    ]


def test_inspect_keeps_unknown_sizes_and_ignores_other_files():
    info = SimpleNamespace(
        id="org/model",
        sha=None,
        gated=None,
        card_data=None,
        siblings=[
            SimpleNamespace(path="model.unknown.gguf", size=None, lfs=None),
            SimpleNamespace(path="weights.bin", size=1_000, lfs=None),
        ],
    )

    result = HuggingFaceDiscovery(
        api=SimpleNamespace(model_info=lambda *args, **kwargs: info)
    ).inspect("org/model", revision="dev")

    assert result.revision == "dev"
    assert result.license is None
    assert result.gated is None
    assert len(result.variants) == 1
    assert result.variants[0].size_bytes is None
    assert result.variants[0].quantization is None


def test_inspect_rejects_empty_repo_id_and_wraps_api_errors():
    discovery = HuggingFaceDiscovery(api=SimpleNamespace())
    with pytest.raises(ValueError, match="repo_id"):
        discovery.inspect("  ")

    def fail(*args, **kwargs):
        raise RuntimeError("403")

    failing = HuggingFaceDiscovery(api=SimpleNamespace(model_info=fail))
    with pytest.raises(DiscoveryError, match="inspection failed"):
        failing.inspect("org/model")
