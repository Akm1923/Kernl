from typing import ClassVar

from kernl.models.discovery import HuggingFaceDiscovery


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
