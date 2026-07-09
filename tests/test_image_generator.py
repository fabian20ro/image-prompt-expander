"""Tests for the fixed ERNIE-Image-Turbo generator."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from image_generator import (
    GUIDANCE,
    INFERENCE_STEPS,
    MODEL_NAME,
    QUANTIZATION,
    _get_model,
    _model_cache,
    clear_model_cache,
    generate_image,
)


def test_fixed_model_contract():
    assert MODEL_NAME == "ernie-image-turbo"
    assert QUANTIZATION == 4
    assert INFERENCE_STEPS == 8
    assert GUIDANCE == 1.0


def test_clear_model_cache():
    _model_cache[False] = object()
    clear_model_cache()
    assert not _model_cache


def test_get_model_cache_hit():
    cached = MagicMock()
    _model_cache[True] = cached
    assert _get_model(tiled_vae=True) is cached
    clear_model_cache()


@patch("image_generator.settings")
@patch("image_generator._model_cache", {})
def test_get_model_loads_local_ernie_q4(mock_settings, temp_dir):
    model_path = temp_dir / "ernie-q4"
    model_path.mkdir()
    mock_settings.image_generation.model_path = model_path

    config_module = ModuleType("mflux.models.common.config")
    config_module.ModelConfig = MagicMock()
    model_config = config_module.ModelConfig.ernie_image_turbo.return_value
    ernie_module = ModuleType("mflux.models.ernie_image")
    instance = MagicMock()
    ernie_module.ErnieImage = MagicMock(return_value=instance)

    with patch.dict(sys.modules, {
        "mflux.models.common.config": config_module,
        "mflux.models.ernie_image": ernie_module,
    }):
        result = _get_model()

    ernie_module.ErnieImage.assert_called_once_with(
        model_config=model_config,
        model_path=str(model_path),
        quantize=None,
    )
    assert result is instance


@patch("image_generator.settings")
@patch("image_generator._model_cache", {})
def test_get_model_requires_provisioned_checkpoint(mock_settings, temp_dir):
    mock_settings.image_generation.model_path = temp_dir / "missing"
    with pytest.raises(FileNotFoundError, match="ERNIE q4 model not found"):
        _get_model()


@patch("image_generator.settings")
@patch("image_generator._model_cache", {})
def test_get_model_applies_tiling_when_tiled_vae(mock_settings, temp_dir):
    model_path = temp_dir / "ernie-q4"
    model_path.mkdir()
    mock_settings.image_generation.model_path = model_path

    config_module = ModuleType("mflux.models.common.config")
    config_module.ModelConfig = MagicMock()
    ernie_module = ModuleType("mflux.models.ernie_image")
    instance = MagicMock()
    ernie_module.ErnieImage = MagicMock(return_value=instance)

    tiling_module = ModuleType("mflux.models.common.vae.tiling_config")
    TilingConfig = MagicMock()
    tiling_module.TilingConfig = TilingConfig

    with patch.dict(sys.modules, {
        "mflux.models.common.config": config_module,
        "mflux.models.ernie_image": ernie_module,
        "mflux.models.common.vae.tiling_config": tiling_module,
    }):
        _get_model(tiled_vae=True)

    assert TilingConfig.called


@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_uses_fixed_parameters(mock_unload, mock_get_model, temp_dir):
    generated = MagicMock()
    model = MagicMock()
    model.generate_image.return_value = generated
    mock_get_model.return_value = model
    output = temp_dir / "image.png"

    generate_image("test prompt", output, seed=42, width=1024, height=768)

    mock_unload.assert_called_once_with()
    model.generate_image.assert_called_once_with(
        seed=42,
        prompt="test prompt",
        num_inference_steps=8,
        guidance=1.0,
        height=768,
        width=1024,
    )
    generated.save.assert_called_once_with(str(output))


@patch("image_generator._get_model")
@patch("image_generator.unload_all_models")
def test_generate_image_random_seed_and_tiling(mock_unload, mock_get_model, temp_dir):
    mock_get_model.return_value.generate_image.return_value = MagicMock()
    with patch("image_generator.random.randint", return_value=123) as randint:
        generate_image("test", temp_dir / "image.png", tiled_vae=True)
    mock_get_model.assert_called_once_with(True)
    mock_unload.assert_called_once_with()
    randint.assert_called_once()
    assert mock_get_model.return_value.generate_image.call_args.kwargs["seed"] == 123


@pytest.mark.parametrize("width,height", [(-1, 512), (512, 0), (511, 512), (512, 513)])
def test_generate_image_rejects_invalid_dimensions(temp_dir, width, height):
    with pytest.raises(ValueError):
        generate_image("test", temp_dir / "image.png", width=width, height=height)
