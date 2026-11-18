from pathlib import Path
from kernl.models.recommendation import recommend
from kernl.models.discovery import ModelCandidate
from kernl.runtime.config import RuntimeConfig
from kernl.runtime.config_store import ConfigStore


def test_recommendation_prefers_compatible_models():
    models = [
        ModelCandidate("bad/model", "bad", 1000, 1, "text-generation", None, (), True),
        ModelCandidate("good/model", "good", 100, 1, "text-generation", None, (), False),
    ]
    result = recommend(models, available_ram_bytes=1_000_000)
    assert result[0].model.repo_id == "good/model"


def test_config_store_round_trip(tmp_path: Path):
    store = ConfigStore(tmp_path)
    config = RuntimeConfig(model_path=tmp_path / "model.gguf")
    store.save("default", config)
    loaded = store.load("default")
    assert loaded.model_path == config.model_path
