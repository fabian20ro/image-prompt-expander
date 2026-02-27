"""Tests for image_enhancer.py - SeedVR2 enhancement wrapper."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_enhancer import (
    clear_enhancer_cache,
    _get_enhancer,
    _enhancer_cache,
    enhance_image,
    collect_images,
)


class TestEnhancerCache:
    """Tests for enhancer caching functionality."""

    def test_clear_enhancer_cache(self):
        """Test that cache clearing works."""
        _enhancer_cache["test_key"] = "test_value"
        assert len(_enhancer_cache) > 0

        clear_enhancer_cache()
        assert len(_enhancer_cache) == 0

    def test_clear_enhancer_cache_calls_gc(self):
        """Test that garbage collection is called on cache clear."""
        import gc
        with patch.object(gc, 'collect') as mock_gc:
            clear_enhancer_cache()
            mock_gc.assert_called_once()


class TestGetEnhancer:
    """Tests for enhancer instantiation."""

    def teardown_method(self):
        """Clear cache after each test."""
        clear_enhancer_cache()

    def test_get_enhancer_cache_hit(self):
        """Test that cached enhancers are reused."""
        mock_enhancer = MagicMock()
        cache_key = (8, True)
        _enhancer_cache[cache_key] = mock_enhancer

        result = _get_enhancer(quantize=8, tiled_vae=True)
        assert result is mock_enhancer

    def test_get_enhancer_mflux_not_installed(self):
        """Test error when mflux is not installed."""
        with patch.dict("sys.modules", {"mflux.models.seedvr2.variants.upscale.seedvr2": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'mflux'")):
                with pytest.raises(ImportError, match="mflux is required"):
                    _get_enhancer(quantize=8)

    @patch("image_enhancer._enhancer_cache", {})
    def test_get_enhancer_creation(self):
        """Test SeedVR2 enhancer creation with mocked mflux."""
        mock_instance = MagicMock()

        module_seedvr2 = ModuleType("mflux.models.seedvr2.variants.upscale.seedvr2")
        module_seedvr2.SeedVR2 = MagicMock(return_value=mock_instance)

        with patch.dict(sys.modules, {
            "mflux.models.seedvr2.variants.upscale.seedvr2": module_seedvr2,
        }):
            result = _get_enhancer(quantize=8, tiled_vae=True)

        module_seedvr2.SeedVR2.assert_called_once_with(quantize=8)
        assert result is mock_instance

    @patch("image_enhancer._enhancer_cache", {})
    def test_get_enhancer_disables_tiling(self):
        """Test that tiled_vae=False disables tiling."""
        mock_instance = MagicMock()
        mock_instance.tiling_config = MagicMock()

        module_seedvr2 = ModuleType("mflux.models.seedvr2.variants.upscale.seedvr2")
        module_seedvr2.SeedVR2 = MagicMock(return_value=mock_instance)

        with patch.dict(sys.modules, {
            "mflux.models.seedvr2.variants.upscale.seedvr2": module_seedvr2,
        }):
            result = _get_enhancer(quantize=8, tiled_vae=False)

        assert result.tiling_config is None


class TestEnhanceImage:
    """Tests for single image enhancement."""

    def teardown_method(self):
        """Clear cache after each test."""
        clear_enhancer_cache()

    def test_enhance_image_file_not_found(self, temp_dir):
        """Test error when image doesn't exist."""
        missing = temp_dir / "nonexistent.png"
        with pytest.raises(FileNotFoundError, match="Image not found"):
            enhance_image(image_path=missing, output_path=missing)

    @patch("image_enhancer._get_enhancer")
    def test_enhance_image_basic(self, mock_get_enhancer, temp_dir):
        """Test basic image enhancement."""
        # Create a real image file
        img = Image.new("RGB", (100, 100), color="red")
        image_path = temp_dir / "test.png"
        img.save(image_path)
        output_path = temp_dir / "enhanced.png"

        mock_enhancer = MagicMock()
        mock_result = MagicMock()
        mock_enhancer.generate_image.return_value = mock_result
        mock_get_enhancer.return_value = mock_enhancer

        # Mock ScaleFactor import
        mock_scale_factor = MagicMock()
        with patch.dict(sys.modules, {
            "mflux.utils.scale_factor": MagicMock(ScaleFactor=mock_scale_factor),
        }):
            result = enhance_image(
                image_path=image_path,
                output_path=output_path,
                softness=0.7,
                seed=42,
                quantize=8,
            )

        assert result == output_path
        mock_enhancer.generate_image.assert_called_once()
        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert call_kwargs["seed"] == 42
        assert call_kwargs["softness"] == 0.7
        assert call_kwargs["image_path"] == str(image_path)
        mock_result.save.assert_called_once_with(path=str(output_path), overwrite=True)

    @patch("image_enhancer._get_enhancer")
    def test_enhance_image_random_seed(self, mock_get_enhancer, temp_dir):
        """Test that a random seed is used when none specified."""
        img = Image.new("RGB", (100, 100))
        image_path = temp_dir / "test.png"
        img.save(image_path)

        mock_enhancer = MagicMock()
        mock_enhancer.generate_image.return_value = MagicMock()
        mock_get_enhancer.return_value = mock_enhancer

        with patch.dict(sys.modules, {
            "mflux.utils.scale_factor": MagicMock(),
        }):
            enhance_image(image_path=image_path, output_path=image_path)

        call_kwargs = mock_enhancer.generate_image.call_args.kwargs
        assert isinstance(call_kwargs["seed"], int)

    @patch("image_enhancer._get_enhancer")
    def test_enhance_image_creates_output_dir(self, mock_get_enhancer, temp_dir):
        """Test that output directory is created."""
        img = Image.new("RGB", (100, 100))
        image_path = temp_dir / "test.png"
        img.save(image_path)
        output_path = temp_dir / "nested" / "dir" / "enhanced.png"

        mock_enhancer = MagicMock()
        mock_enhancer.generate_image.return_value = MagicMock()
        mock_get_enhancer.return_value = mock_enhancer

        with patch.dict(sys.modules, {
            "mflux.utils.scale_factor": MagicMock(),
        }):
            enhance_image(image_path=image_path, output_path=output_path)

        assert output_path.parent.exists()


class TestCollectImages:
    """Tests for image collection from various sources."""

    def test_collect_single_file(self, temp_dir):
        """Test collecting a single image file."""
        img = Image.new("RGB", (10, 10))
        image_path = temp_dir / "test.png"
        img.save(image_path)

        result = collect_images(str(image_path))
        assert result == [image_path]

    def test_collect_single_file_not_image(self, temp_dir):
        """Test error for non-image file."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("not an image")

        with pytest.raises(ValueError, match="Not an image file"):
            collect_images(str(text_file))

    def test_collect_directory(self, temp_dir):
        """Test collecting all images from a directory."""
        for name in ["a.png", "b.jpg", "c.txt"]:
            p = temp_dir / name
            if name.endswith((".png", ".jpg")):
                Image.new("RGB", (10, 10)).save(p)
            else:
                p.write_text("text")

        result = collect_images(str(temp_dir))
        names = [r.name for r in result]
        assert "a.png" in names
        assert "b.jpg" in names
        assert "c.txt" not in names

    def test_collect_empty_directory(self, temp_dir):
        """Test error for directory with no images."""
        empty = temp_dir / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match="No images found in directory"):
            collect_images(str(empty))

    def test_collect_glob_pattern(self, temp_dir):
        """Test collecting images via glob pattern."""
        for name in ["a.png", "b.png", "c.jpg"]:
            Image.new("RGB", (10, 10)).save(temp_dir / name)

        result = collect_images(str(temp_dir / "*.png"))
        assert len(result) == 2

    def test_collect_glob_no_match(self, temp_dir):
        """Test error when glob matches no images."""
        with pytest.raises(ValueError, match="No images found matching pattern"):
            collect_images(str(temp_dir / "*.xyz"))
