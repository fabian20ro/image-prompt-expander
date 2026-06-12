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

    def test_clear_model_cache_calls_gc(self="called_once"):
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
    def test_get_model_config_selection(self):
        """Verify that _get_model selects the correct ModelConfig method for each model."""
        mock_config_module = ModuleType("mflux.models.common.config.model_config")
        mock_config_module.ModelConfig = MagicMock()
        mock_config_module.ModelConfig.z_image_turbo.return_value = MagicMock()
        mock_config_module.ModelConfig.flux2_klein_4b.return_value = MagicMock()
        mock_config_module.ModelConfig.flux2_klein_9b.return_value = MagicMock()

        mock_z_image = ModuleType("mflux.models.z_image")
        mock_z_image.ZImageTurbo = MagicMock()

        mock_flux2 = ModuleType("mflux.models.flux2")
        mock_flux2.Flux2Klein = MagicMock()

        with patch.dict(sys.modules, {
            "mflux.models.common.config.model_config": mock_config_module,
            "mflux.models.z_image": mock_z_image,
            "mflux.models.flux2": mock_flux2,
        }):
            # Test z-image-turbo
            _get_model("z-image-turbo", quantize=8)
            mock_config_module.ModelConfig.z_image_turbo.assert_called_once()

            # Test flux2-klein-4b
            _get_model("flux2-klein-4b", quantize=4)
            mock_config_module.ModelConfig.flux2_klein_4b.assert_called_once()

            # Test flux2-klein-9b
            _get_model("flux2-klein-9b", quantize=4)
            mock_config_module.ModelConfig.flux2_klein_9b.assert_called_once()

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
    def test_generate_image_invalid_dimensions(self, mock_get_model, temp_dir):
        """Test errors for invalid width and height."""
        mock_flux = MagicMock()
        mock_get_model.return_value = mock_flux
        output_path = temp_dir / "test.png"

        # Test non-positive dimensions
        with pytest.raises(ValueError, match="must be positive"):
            generate_image(prompt="test", output_path=output_path, width=0, height=1024)
        with pytest.raises(ValueError, match="must be positive"):
            generate_image(prompt="test", output_path=output_path, width=1024, height=-1)

        # Test non-multiples of 8
        with pytest.raises(ValueError, match="must be multiples of 8"):
            generate_image(prompt="test", output_path=output_path, width=1023, height=1024)
        with pytest.raises(ValueError, match="must be multiples of 8"):
            generate_image(prompt="test", output_path=output_path, width=1024, height=1025)

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

class TestImageGenerationValidation:
    """Tests for dimension validation in generate_image."""

    @patch("image_generator._get_model")
    def test_invalid_width_or_height(self, mock_get_model, temp_dir):
        """Test that invalid dimensions raise ValueError."""
        mock_flux = MagicMock()
        mock_get_model.return_value = mock_flux
        output_path = temp_dir / "test.png"
        
        with pytest.raises(ValueError, match="Width and height must be positive"):
            generate_image("prompt", output_path, width=-1)
        with pytest.raises(ValueError, match="Width and height must be positive"):
            generate_image("prompt", output_path, height=0)
        with pytest.raises(ValueError, match="Width and height must be multiples of 8"):
            generate_image("prompt", output_path, width=7)

    @patch("image_generator._get_model")
    def test_valid_dimensions(self, mock_get_model, temp_dir):
        """Test that valid dimensions pass."""
        mock_flux = MagicMock()
        mock_get_model.return_value = mock_flux
        output_path = temp_dir / "test.png"
        generate_image("prompt", output_path, width=1024, height=1024)
        generate_image("prompt", output_path, width=512, height=512)


    @patch("image_generator._get_model")
    def test_tiled_vae_flag_passed(self, mock_get_model, temp_dir):
        """Test that tiled_vae flag is passed to _get_model."""
        mock_flux = MagicMock()
        mock_get_model.return_value = mock_flux
        
        output_path = temp_dir / "test.png"
        generate_image(prompt="test", output_path=output_path, tiled_vae=True)
        
        args, kwargs = mock_get_model.call_args
        # Check if it's in kwargs or args
        tiled_vae_val = kwargs.get("tiled_vae", args[2] if len(args) > 2 else None)
        assert tiled_vae_val is True


