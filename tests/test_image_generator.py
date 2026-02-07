"""Tests for image_generator.py - mflux wrapper for Apple Silicon."""

import random
import sys
from types import ModuleType
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image_generator import (
    MODEL_DEFAULTS,
    MODEL_PATHS,
    SUPPORTED_MODELS,
    generate_image,
    generate_images_for_prompts,
    clear_model_cache,
    _get_model,
    _model_cache,
)


class TestModelConstants:
    """Tests for model configuration constants."""

    def test_model_defaults_has_all_supported_models(self):
        """All supported models should have defaults."""
        for model in SUPPORTED_MODELS:
            assert model in MODEL_DEFAULTS
            assert "steps" in MODEL_DEFAULTS[model]

    def test_model_paths_has_all_supported_models(self):
        """All supported models should have a path entry (can be None)."""
        for model in SUPPORTED_MODELS:
            assert model in MODEL_PATHS

    def test_supported_models_list(self):
        """Supported models should include expected models."""
        assert "z-image-turbo" in SUPPORTED_MODELS
        assert len(SUPPORTED_MODELS) >= 1


class TestModelCache:
    """Tests for model caching functionality."""

    def test_clear_model_cache(self):
        """Test that cache clearing works."""
        # Manually populate cache
        _model_cache["test_key"] = "test_value"
        assert len(_model_cache) > 0

        clear_model_cache()
        assert len(_model_cache) == 0

    def test_clear_model_cache_calls_gc(self):
        """Test that garbage collection is called on cache clear."""
        import gc
        with patch.object(gc, 'collect') as mock_gc:
            clear_model_cache()
            mock_gc.assert_called_once()


class TestGetModel:
    """Tests for model instantiation."""

    def teardown_method(self):
        """Clear cache after each test."""
        clear_model_cache()

    @patch("image_generator._model_cache", {})
    def test_get_model_unsupported(self):
        """Test error for unsupported model."""
        with pytest.raises(ValueError, match="Unsupported model"):
            _get_model("nonexistent-model", quantize=8)

    @patch("image_generator._model_cache", {})
    def test_get_model_mflux_not_installed(self):
        """Test error when mflux is not installed."""
        with patch.dict("sys.modules", {"mflux.models.common.config.model_config": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'mflux'")):
                with pytest.raises(ImportError, match="mflux is required"):
                    _get_model("z-image-turbo", quantize=8)

    def test_get_model_cache_hit(self):
        """Test that cached models are reused."""
        mock_model = MagicMock()
        cache_key = ("z-image-turbo", 8, True)
        _model_cache[cache_key] = mock_model

        result = _get_model("z-image-turbo", quantize=8, tiled_vae=True)
        assert result is mock_model

    @patch("image_generator._model_cache", {})
    def test_get_model_z_image_turbo_creation(self):
        """Test z-image-turbo model creation."""
        mock_instance = MagicMock()
        mock_tiling_instance = MagicMock()

        # Build lightweight fake mflux module tree so importing does not
        # initialize native MLX/Metal components in tests.
        module_model_config = ModuleType("mflux.models.common.config.model_config")
        module_model_config.ModelConfig = MagicMock()
        module_model_config.ModelConfig.z_image_turbo.return_value = MagicMock()

        module_z_image = ModuleType("mflux.models.z_image")
        module_z_image.ZImageTurbo = MagicMock(return_value=mock_instance)

        module_tiling = ModuleType("mflux.models.common.vae.tiling_config")
        module_tiling.TilingConfig = MagicMock(return_value=mock_tiling_instance)

        with patch.dict(sys.modules, {
            "mflux.models.common.config.model_config": module_model_config,
            "mflux.models.z_image": module_z_image,
            "mflux.models.common.vae.tiling_config": module_tiling,
        }):
            result = _get_model("z-image-turbo", quantize=8, tiled_vae=True)

        module_z_image.ZImageTurbo.assert_called_once()
        module_tiling.TilingConfig.assert_called_once()
        assert result is mock_instance


class TestGenerateImage:
    """Tests for single image generation."""

    def teardown_method(self):
        """Clear cache after each test."""
        clear_model_cache()

    def test_generate_image_unsupported_model(self, temp_dir):
        """Test error for unsupported model."""
        output_path = temp_dir / "test.png"
        with pytest.raises(ValueError, match="Unsupported model"):
            generate_image(
                prompt="test",
                output_path=output_path,
                model="nonexistent-model",
            )

    @patch("image_generator._get_model")
    def test_generate_image_default_steps(self, mock_get_model, temp_dir):
        """Test that default steps are used when not specified."""
        mock_flux = MagicMock()
        mock_get_model.return_value = mock_flux

        output_path = temp_dir / "test.png"
        generate_image(prompt="test", output_path=output_path, model="z-image-turbo")

        # Check that default steps (9 for z-image-turbo) were used
        call_args = mock_flux.generate_image.call_args
        assert call_args.kwargs.get("num_inference_steps") == 9

    @patch("image_generator._get_model")
    def test_generate_image_custom_steps(self, mock_get_model, temp_dir):
        """Test that custom steps override defaults."""
        mock_flux = MagicMock()
        mock_get_model.return_value = mock_flux

        output_path = temp_dir / "test.png"
        generate_image(prompt="test", output_path=output_path, steps=15)

        call_args = mock_flux.generate_image.call_args
        assert call_args.kwargs.get("num_inference_steps") == 15

    @patch("image_generator._get_model")
    @patch("image_generator.random.randint")
    def test_generate_image_random_seed(self, mock_randint, mock_get_model, temp_dir):
        """Test that random seed is generated when not specified."""
        mock_randint.return_value = 12345
        mock_flux = MagicMock()
        mock_get_model.return_value = mock_flux

        output_path = temp_dir / "test.png"
        generate_image(prompt="test", output_path=output_path, seed=None)

        call_args = mock_flux.generate_image.call_args
        assert call_args.kwargs.get("seed") == 12345
        mock_randint.assert_called_once()

    @patch("image_generator._get_model")
    def test_generate_image_specified_seed(self, mock_get_model, temp_dir):
        """Test that specified seed is used."""
        mock_flux = MagicMock()
        mock_get_model.return_value = mock_flux

        output_path = temp_dir / "test.png"
        generate_image(prompt="test", output_path=output_path, seed=42)

        call_args = mock_flux.generate_image.call_args
        assert call_args.kwargs.get("seed") == 42

    @patch("image_generator._get_model")
    def test_generate_image_creates_output_dir(self, mock_get_model, temp_dir):
        """Test that output directory is created if needed."""
        mock_flux = MagicMock()
        mock_get_model.return_value = mock_flux

        nested_path = temp_dir / "nested" / "dir" / "test.png"
        generate_image(prompt="test", output_path=nested_path)

        assert nested_path.parent.exists()

    @patch("image_generator._get_model")
    def test_generate_image_z_image_turbo_saves_directly(self, mock_get_model, temp_dir):
        """Test that z-image-turbo saves PIL image directly."""
        mock_pil_image = MagicMock()
        mock_flux = MagicMock()
        mock_flux.generate_image.return_value = mock_pil_image
        mock_get_model.return_value = mock_flux

        output_path = temp_dir / "test.png"
        generate_image(prompt="test", output_path=output_path, model="z-image-turbo")

        # z-image-turbo returns PIL Image, should call .save()
        mock_pil_image.save.assert_called_once_with(str(output_path))

    @patch("image_generator._get_model")
    def test_generate_image_flux2_uses_result_save(self, mock_get_model, temp_dir):
        """Test that flux2 models use GeneratedImage.save()."""
        mock_result = MagicMock()
        mock_flux = MagicMock()
        mock_flux.generate_image.return_value = mock_result
        mock_get_model.return_value = mock_flux

        output_path = temp_dir / "test.png"
        generate_image(prompt="test", output_path=output_path, model="flux2-klein-4b")

        # flux2 returns GeneratedImage, should call .save(path=...)
        mock_result.save.assert_called_once_with(path=str(output_path))


class TestGenerateImagesForPrompts:
    """Tests for batch image generation."""

    def teardown_method(self):
        """Clear cache after each test."""
        clear_model_cache()

    @patch("image_generator.generate_image")
    def test_generate_images_for_prompts_basic(self, mock_gen, temp_dir):
        """Test basic batch generation."""
        prompts = ["prompt 1", "prompt 2", "prompt 3"]

        result = generate_images_for_prompts(
            prompts=prompts,
            output_dir=temp_dir,
            prefix="test",
            images_per_prompt=1,
        )

        assert len(result) == 3
        assert mock_gen.call_count == 3

    @patch("image_generator.generate_image")
    def test_generate_images_for_prompts_multiple_per_prompt(self, mock_gen, temp_dir):
        """Test multiple images per prompt."""
        prompts = ["prompt 1", "prompt 2"]

        result = generate_images_for_prompts(
            prompts=prompts,
            output_dir=temp_dir,
            prefix="test",
            images_per_prompt=2,
        )

        assert len(result) == 4  # 2 prompts * 2 images
        assert mock_gen.call_count == 4

    @patch("image_generator.generate_image")
    def test_generate_images_for_prompts_max_prompts(self, mock_gen, temp_dir):
        """Test max_prompts limit."""
        prompts = ["prompt 1", "prompt 2", "prompt 3", "prompt 4"]

        result = generate_images_for_prompts(
            prompts=prompts,
            output_dir=temp_dir,
            prefix="test",
            max_prompts=2,
        )

        assert len(result) == 2
        assert mock_gen.call_count == 2

    @patch("image_generator.generate_image")
    def test_generate_images_for_prompts_seed_increment(self, mock_gen, temp_dir):
        """Test that seed increments for each image."""
        prompts = ["prompt 1", "prompt 2"]

        generate_images_for_prompts(
            prompts=prompts,
            output_dir=temp_dir,
            prefix="test",
            seed=100,
        )

        # Check seeds used: 100, 101
        calls = mock_gen.call_args_list
        assert calls[0].kwargs.get("seed") == 100
        assert calls[1].kwargs.get("seed") == 101

    @patch("image_generator.generate_image")
    def test_generate_images_for_prompts_progress_callback(self, mock_gen, temp_dir):
        """Test progress callback is called."""
        prompts = ["prompt 1", "prompt 2"]
        progress_calls = []

        generate_images_for_prompts(
            prompts=prompts,
            output_dir=temp_dir,
            prefix="test",
            on_progress=lambda p, i, t: progress_calls.append((p, i, t)),
        )

        assert len(progress_calls) == 2
        assert progress_calls[0] == (0, 0, 2)  # First image
        assert progress_calls[1] == (1, 0, 2)  # Second image

    @patch("image_generator.generate_image")
    def test_generate_images_for_prompts_creates_dir(self, mock_gen, temp_dir):
        """Test that output directory is created."""
        prompts = ["prompt"]
        nested_dir = temp_dir / "nested" / "output"

        generate_images_for_prompts(
            prompts=prompts,
            output_dir=nested_dir,
            prefix="test",
        )

        assert nested_dir.exists()
