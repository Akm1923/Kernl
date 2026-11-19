from kernl.models.compatibility import assess_model
from kernl.models.discovery import ModelCandidate


def test_memory_overflow_is_incompatible():
    model = ModelCandidate("org/model", "org", 0, 0, None, None, (), False)
    result = assess_model(model, available_ram_bytes=100, required_ram_bytes=101)
    assert result.compatible is False
    assert "available RAM" in result.reasons[0]


def test_open_model_with_unknown_size_has_unknown_compatibility():
    model = ModelCandidate("org/model", "org", 0, 0, None, None, (), False)
    result = assess_model(model, available_ram_bytes=100)
    assert result.compatible is None
    assert result.score == 0.5


def test_unknown_memory_requirements_are_not_reported_as_compatible():
    model = ModelCandidate("org/model", "org", 0, 0, None, None, (), False)
    result = assess_model(model, available_ram_bytes=1_000_000)
    assert result.compatible is None
    assert any("memory requirements are unknown" in reason for reason in result.reasons)


def test_known_memory_requirement_within_ram_is_compatible():
    model = ModelCandidate("org/model", "org", 0, 0, None, None, (), False)
    result = assess_model(model, available_ram_bytes=1_000, required_ram_bytes=900)
    assert result.compatible is True


def test_negative_memory_inputs_are_rejected():
    model = ModelCandidate("org/model", "org", 0, 0, None, None, (), False)
    for available, required in ((-1, 1), (1, -1)):
        try:
            assess_model(
                model, available_ram_bytes=available, required_ram_bytes=required
            )
        except ValueError:
            continue
        raise AssertionError("negative memory values must be rejected")
