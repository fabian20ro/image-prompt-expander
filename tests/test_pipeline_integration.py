
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from pipeline import (
    PipelineExecutor,
    PipelineResult,
    PipelineConfig,
    ImageGenerationConfig,
    EnhancementConfig,
)


@pytest.fixture
def mock_grammar_file(tmp_path):
    grammar_file = tmp_path / "test.tracery.json"
    grammar_file.write_text('{"root": {"_repeat": ["A"]}}')
    meta_file = tmp_path / "test.metaprompt.json"
    meta_file.write_text('{"user_prompt": "a dragon"}')
    return grammar_file


def test_run_from_grammar_success(mock_grammar_file, tmp_path):
    executor = PipelineExecutor()

    with patch("pipeline.generate_grammar", return_value=('{ "root": {"_repeat": ["A"]}}', False, None)):
        with patch("pipeline.PipelineExecutor.run_from_grammar_text", return_value=PipelineResult(success=True, run_id="test_id")):
            result = executor.run_from_grammar(mock_grammar_file)
            assert result.success is True
            assert result.run_id == "test_id"


def test_run_from_grammar_no_meta(mock_grammar_file):
    executor = PipelineExecutor()

    with patch("pipeline.generate_grammar", return_value=('{ "root": {"_repeat": ["A"]}}', False, None)):
        with patch("pipeline.PipelineExecutor.run_from_grammar_text", return_value=PipelineResult(success=True, run_id="test_id")):
            result = executor.run_from_grammar(mock_grammar_file)
            assert result.success is True
            assert result.run_id == "test_id"


def test_run_from_grammar_text_invalid_grammar_returns_error(tmp_path):
    """Invalid Tracery grammar should produce a failed PipelineResult with an error message."""
    executor = PipelineExecutor()

    invalid_grammar = '{"root": {invalid json}}'

    with patch("pipeline.create_gallery") as mock_gallery, \
         patch("pipeline.generate_master_index") as mock_index, \
         patch("pipeline.append_grammar_revision"):
        result = executor.run_from_grammar_text(
            grammar=invalid_grammar,
            count=3,
            output_dir=tmp_path / "output",
        )

    assert result.success is False
    assert result.prompt_count == 0
    assert result.image_count == 0
    assert isinstance(result.error, str)
    assert "Tracery expansion failed" in result.error or "Invalid JSON grammar" in result.error


def test_run_from_grammar_text_progress_callback_invoked(tmp_path):
    """Pipeline should invoke the progress callback with expected stages on success."""

    recorded_stages = []

    def track_progress(stage, current=0, total=0, message=""):
        recorded_stages.append(stage)

    executor = PipelineExecutor(on_progress=track_progress)

    grammar = json.dumps({"origin": ["a dragon"]})

    with patch("pipeline.create_gallery") as mock_gallery, \
         patch("pipeline.generate_master_index") as mock_index, \
         patch("pipeline.append_grammar_revision"):
        result = executor.run_from_grammar_text(
            grammar=grammar,
            count=2,
            output_dir=tmp_path / "output",
        )

    assert result.success is True
    assert "expanding_prompts" in recorded_stages
    # The pipeline should report 0 then count for the expanding_prompts stage
    expand_indices = [i for i, s in enumerate(recorded_stages) if s == "expanding_prompts"]
    assert len(expand_indices) >= 1


def test_run_full_pipeline_grammar_failure_returns_error(tmp_path):
    """Pipeline should return failure result when grammar generation fails."""

    executor = PipelineExecutor()

    with patch("pipeline.generate_grammar", side_effect=Exception("LM unavailable")):
        result = executor.run_full_pipeline(
            prompt="a dragon flying",
            count=10,
            output_dir=tmp_path / "output",
        )

    assert result.success is False
    assert isinstance(result.error, str)
    assert "Grammar generation failed" in result.error


def test_run_from_grammar_text_empty_expansion(tmp_path):
    """Zero-prompt expansion should produce a successful result with zero prompts."""

    recorded_stages = []

    def track_progress(stage, current=0, total=0, message=""):
        recorded_stages.append(stage)

    executor = PipelineExecutor(on_progress=track_progress)

    grammar = json.dumps({"origin": ["test"]})

    output_dir = tmp_path / "output"

    with patch("pipeline.create_gallery") as mock_gallery, \
         patch("pipeline.generate_master_index") as mock_index, \
         patch("pipeline.append_grammar_revision"), \
         patch("pipeline.run_tracery", return_value=[]):
        result = executor.run_from_grammar_text(
            grammar=grammar,
            count=50,
            output_dir=output_dir,
        )

    assert result.success is True
    assert result.prompt_count == 0
    assert result.image_count == 0
    # Progress callback should still report all expected stages
    assert "expanding_prompts" in recorded_stages


def test_run_from_grammar_text_metadata_structure(tmp_path):
    """run_from_grammar_text should produce correct metadata structure."""

    output_dir = tmp_path / "output"

    with patch("pipeline.create_gallery"), \
         patch("pipeline.generate_master_index"):
        result = PipelineExecutor().run_from_grammar_text(
            grammar='{"origin": ["dragon"]}',
            count=1,
            output_dir=output_dir,
            user_prompt="test prompt",
            source="manual",
            display_title="custom title",
        )

    assert result.success is True
    metadata_file = output_dir / "image.metaprompt.json"
    import json as _json
    metadata = _json.loads(metadata_file.read_text())
    assert metadata["source"] == "manual"
    assert metadata["user_prompt"] == "test prompt"
    assert metadata["display_title"] == "custom title"


def test_pipeline_config_to_dict_serialization():
    """PipelineConfig.to_dict() should convert all fields correctly."""

    config = PipelineConfig(
        prompt="test",
        count=10,
        prefix="img",
        temperature=0.8,
        no_cache=True,
        image=ImageGenerationConfig(enabled=True, width=1024, height=1024),
        enhancement=EnhancementConfig(enabled=False),
    )

    d = config.to_dict()
    assert d["prompt"] == "test"
    assert d["count"] == 10
    assert d["prefix"] == "img"
    assert d["temperature"] == 0.8
    assert d["no_cache"] is True
    assert d["image"]["enabled"] is True
    assert d["image"]["width"] == 1024
