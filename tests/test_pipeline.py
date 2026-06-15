"""Tests for pipeline executor - the core orchestration layer."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from conftest import create_run_files
from pipeline import (
    PipelineExecutor,
    PipelineResult,
    PipelineConfig,
    ImageGenerationConfig,
    EnhancementConfig,
)


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_successful_result(self):
        result = PipelineResult(
            success=True,
            run_id="test_run",
            output_dir=Path("/tmp/test"),
            prompt_count=10,
            image_count=5,
        )
        assert result.success is True
        assert result.run_id == "test_run"
        assert result.error is None

    def test_failed_result(self):
        result = PipelineResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.run_id is None


class TestImageGenerationConfig:
    """Tests for ImageGenerationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ImageGenerationConfig()
        assert config.enabled is False
        assert config.images_per_prompt == 1
        assert config.width == 864
        assert config.height == 1152
        assert config.steps is None
        assert config.quantize == 8
        assert config.seed is None
        assert config.tiled_vae is False
        assert config.resume is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ImageGenerationConfig(
            enabled=True,
            width=1024,
            height=1024,
            steps=30,
            seed=42,
        )
        assert config.enabled is True
        assert config.width == 1024
        assert config.steps == 30
        assert config.seed == 42

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ImageGenerationConfig(
            enabled=True,
            width=1024,
            height=768,
            steps=25,
        )
        result = config.to_dict()
        assert result["enabled"] is True
        assert result["width"] == 1024
        assert result["height"] == 768
        assert result["steps"] == 25
        # Optional fields should not be present if None
        assert "seed" not in result or result.get("seed") is None
        assert "max_prompts" not in result  # None by default

    def test_to_dict_excludes_none_optionals(self):
        """Test that None optional values are excluded from dict."""
        config = ImageGenerationConfig()
        result = config.to_dict()
        assert "steps" not in result  # None by default
        assert "seed" not in result  # None by default
        assert "max_prompts" not in result  # None by default


class TestEnhancementConfig:
    """Tests for EnhancementConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EnhancementConfig()
        assert config.enabled is False
        assert config.softness == 0.5
        assert config.batch_after is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EnhancementConfig(
            enabled=True,
            softness=0.3,
            batch_after=True,
        )
        assert config.enabled is True
        assert config.softness == 0.3
        assert config.batch_after is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = EnhancementConfig(enabled=True, softness=0.7)
        result = config.to_dict()
        assert result["enabled"] is True
        assert result["softness"] == 0.7
        assert result["batch_after"] is False


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.prompt == ""
        assert config.count == 50
        assert config.prefix == "image"
        assert config.model == "flux2-klein-4b"
        assert config.temperature == 0.7
        assert config.no_cache is False
        assert isinstance(config.image, ImageGenerationConfig)
        assert isinstance(config.enhancement, EnhancementConfig)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            prompt="a dragon flying",
            count=100,
            prefix="dragon",
            model="z-image",
            image=ImageGenerationConfig(enabled=True, width=1024),
            enhancement=EnhancementConfig(enabled=True),
        )
        assert config.prompt == "a dragon flying"
        assert config.count == 100
        assert config.image.enabled is True
        assert config.image.width == 1024
        assert config.enhancement.enabled is True

    def test_from_kwargs_basic(self):
        """Test creating config from flat kwargs."""
        config = PipelineConfig.from_kwargs(
            prompt="test prompt",
            count=25,
            prefix="test",
        )
        assert config.prompt == "test prompt"
        assert config.count == 25
        assert config.prefix == "test"

    def test_from_kwargs_with_image_params(self):
        """Test creating config from kwargs with image generation params."""
        config = PipelineConfig.from_kwargs(
            prompt="test",
            generate_images=True,
            width=1024,
            height=768,
            steps=30,
            seed=42,
        )
        assert config.image.enabled is True
        assert config.image.width == 1024
        assert config.image.height == 768
        assert config.image.steps == 30
        assert config.image.seed == 42
        assert config.image.max_prompts is None

    def test_from_kwargs_with_enhancement_params(self):
        """Test creating config from kwargs with enhancement params."""
        config = PipelineConfig.from_kwargs(
            prompt="test",
            enhance=True,
            enhance_softness=0.3,
            enhance_after=True,
        )
        assert config.enhancement.enabled is True
        assert config.enhancement.softness == 0.3
        assert config.enhancement.batch_after is True

    def test_from_kwargs_full_example(self):
        """Test creating config with all old-style params."""
        config = PipelineConfig.from_kwargs(
            prompt="a dragon",
            count=50,
            prefix="dragon",
            model="z-image",
            temperature=0.8,
            no_cache=True,
            generate_images=True,
            images_per_prompt=2,
            width=1024,
            height=1024,
            steps=25,
            quantize=4,
            seed=123,
            max_prompts=10,
            tiled_vae=False,
            enhance=True,
            enhance_softness=0.4,
            enhance_after=True,
            resume=True,
        )
        # Main config
        assert config.prompt == "a dragon"
        assert config.count == 50
        assert config.model == "z-image"
        assert config.temperature == 0.8
        assert config.no_cache is True

        # Image config
        assert config.image.enabled is True
        assert config.image.images_per_prompt == 2
        assert config.image.width == 1024
        assert config.image.steps == 25
        assert config.image.quantize == 4
        assert config.image.seed == 123
        assert config.image.max_prompts == 10
        assert config.image.tiled_vae is False
        assert config.image.resume is True

        # Enhancement config
        assert config.enhancement.enabled is True
        assert config.enhancement.softness == 0.4
        assert config.enhancement.batch_after is True


class TestPipelineExecutorInit:
    """Tests for PipelineExecutor initialization."""

    def test_init_with_callbacks(self):
        progress_cb = MagicMock()
        image_cb = MagicMock()
        executor = PipelineExecutor(on_progress=progress_cb, on_image_ready=image_cb)
        assert executor.on_progress == progress_cb
        assert executor.on_image_ready == image_cb

    def test_init_without_callbacks(self):
        executor = PipelineExecutor()
        # Should use null callbacks
        executor.on_progress("test", 0, 0, "message")  # Should not raise
        executor.on_image_ready("run_id", "filename")  # Should not raise


class TestRunFullPipeline:
    """Tests for run_full_pipeline method."""

    @patch("pipeline.generate_grammar")
    @patch("pipeline.run_tracery")
    @patch("pipeline.create_gallery")
    @patch("pipeline.generate_master_index")
    def test_run_full_pipeline_prompts_only(
        self, mock_index, mock_gallery, mock_tracery, mock_grammar, temp_dir
    ):
        """Test full pipeline without image generation."""
        mock_grammar.return_value = ('{"origin": ["test"]}', False, "raw response")
        mock_tracery.return_value = ["prompt 1", "prompt 2", "prompt 3"]
        mock_gallery.return_value = temp_dir / "test_gallery.html"

        # Mock paths
        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            progress_calls = []
            executor = PipelineExecutor(
                on_progress=lambda s, c, t, m: progress_calls.append((s, c, t, m))
            )

            result = executor.run_full_pipeline(
                prompt="test prompt",
                count=3,
                prefix="test",
                generate_images=False,
            )

        assert result.success is True
        assert result.prompt_count == 3
        assert result.image_count == 0
        assert result.output_dir is not None

        # Check progress was reported
        assert any(s == "generating_grammar" for s, _, _, _ in progress_calls)
        assert any(s == "expanding_prompts" for s, _, _, _ in progress_calls)

        # Check files were created
        assert result.output_dir.exists()
        assert (result.output_dir / "test.metaprompt.json").exists()

    @patch("pipeline.generate_grammar")
    def test_run_full_pipeline_grammar_failure(self, mock_grammar, temp_dir):
        """Test handling of grammar generation failure."""
        from tracery_runner import TraceryError
        mock_grammar.side_effect = Exception("LM Studio not available")

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"

            executor = PipelineExecutor()
            result = executor.run_full_pipeline(prompt="test", count=1)

        assert result.success is False
        assert "Grammar generation failed" in result.error

    @patch("pipeline.generate_grammar")
    @patch("pipeline.run_tracery")
    def test_run_full_pipeline_tracery_failure(self, mock_tracery, mock_grammar, temp_dir):
        """Test handling of Tracery expansion failure."""
        from tracery_runner import TraceryError
        mock_grammar.return_value = ('{"origin": ["test"]}', False, None)
        mock_tracery.side_effect = TraceryError("Invalid grammar")

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"

            executor = PipelineExecutor()
            result = executor.run_full_pipeline(prompt="test", count=1)

        assert result.success is False
        assert "Tracery expansion failed" in result.error


class TestRunFromGrammar:
    """Tests for run_from_grammar method."""

    @patch("pipeline.run_tracery")
    @patch("pipeline.create_gallery")
    @patch("pipeline.generate_master_index")
    def test_run_from_grammar_success(
        self, mock_index, mock_gallery, mock_tracery, temp_dir
    ):
        """Test running pipeline from existing grammar file."""
        grammar_dir = temp_dir / "grammars"
        grammar_dir.mkdir()
        grammar_file = grammar_dir / "test.tracery.json"
        grammar_file.write_text('{"origin": ["test"]}')

        mock_tracery.return_value = ["prompt 1"]
        mock_gallery.return_value = temp_dir / "prompts" / "test_gallery.html"

        with patch("pipeline.paths") as mock_paths:
            mock_paths.prompts_dir = temp_dir / "prompts"
            mock_paths.prompts_dir.mkdir(parents=True)
            mock_paths.generated_dir = temp_dir

            executor = PipelineExecutor()
            result = executor.run_from_grammar(grammar_path=grammar_file, count=1)

        assert result.success is True
        assert result.prompt_count == 1
        assert result.output_dir is not None

    def test_run_from_grammar_metadata_match(self, temp_dir):
        """Test that run_from_grammar picks the metadata matching the grammar filename."""
        grammar_dir = temp_dir / "grammars"
        grammar_dir.mkdir()

        meta1 = grammar_dir / "match_me.metaprompt.json"
        meta1.write_text(json.dumps({"user_prompt": "correct"}))

        meta2 = grammar_dir / "wrong.metaprompt.json"
        meta2.write_text(json.dumps({"user_prompt": "wrong"}))

        grammar_file = grammar_dir / "match_me.tracery.json"
        grammar_file.write_text('{"origin": ["test"]}')

        from pipeline import PipelineExecutor
        executor = PipelineExecutor()

        with patch("pipeline.run_tracery") as mock_tracery:
            mock_tracery.return_value = ["prompt 1"]
            with patch("pipeline.create_gallery"):
                with patch("pipeline.generate_master_index"):
                    result = executor.run_from_grammar(grammar_path=grammar_file, count=1)
                    assert result.success is True

    def test_run_from_grammar_metadata_mismatch_fallback(self, temp_dir):
        """Test fallback to any metaprompt in the same directory if no direct match."""
        grammar_dir = temp_dir / "grammars_fallback"
        grammar_dir.mkdir()

        meta1 = grammar_dir / "some_other_file.metaprompt.json"
        meta1.write_text(json.dumps({"user_prompt": "fallback_user"}))

        grammar_file = grammar_dir / "target.tracery.json"
        grammar_file.write_text('{"origin": ["test"]}')

        from pipeline import PipelineExecutor
        executor = PipelineExecutor()

        with patch("pipeline.run_tracery") as mock_tracery:
            mock_tracery.return_value = ["prompt 1"]
            with patch("pipeline.create_gallery"):
                with patch("pipeline.generate_master_index"):
                    result = executor.run_from_grammar(grammar_path=grammar_file, count=1)
                    assert result.success is True
                    assert result.output_dir is not None
