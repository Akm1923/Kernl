from kernl.models.compatibility import assess_model
from kernl.models.discovery import ModelCandidate

def test_memory_overflow_is_incompatible():
    model = ModelCandidate("org/model", "org", 0, 0, None, None, (), False)
    result = assess_model(model, available_ram_bytes=100, required_ram_bytes=101)
    assert result.compatible is False
    assert "available RAM" in result.reasons[0]

def test_open_model_can_be_compatible():
    model = ModelCandidate("org/model", "org", 0, 0, None, None, (), False)
    result = assess_model(model, available_ram_bytes=100)
    assert result.compatible is True
    assert result.score == 1.0
