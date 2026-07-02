"""Tests for centralized configuration."""

import os
from pathlib import Path

import pytest
from unittest.mock import patch

from config import Settings, LMStudioConfig, ImageGenerationConfig, ServerConfig, paths


class TestConfig:
    """Tests for centralized configuration."""

    def test_default_settings(self):
        """Test that default settings are loaded correctly."""
        settings = Settings()

        assert settings.lm_studio.base_url == "http://localhost:1234/v1"
        assert settings.image_generation.default_width == 864
        assert settings.image_generation.default_height == 1152
        assert settings.image_generation.seed == 0
        assert settings.lm_studio.model == "google/gemma-4-26b-a4b-qat"
        assert settings.image_generation.model_path.name == "ernie-image-turbo-4bit"
        assert settings.server.sse_queue_size == 100
        assert settings.enhancement.default_softness == 0.5

    def test_settings_from_env(self):
        """Test that settings can be loaded from environment variables."""
        env_vars = {
            "PROMPT_GEN_LM_STUDIO_URL": "http://test:5000/v1",
            "PROMPT_GEN_DEFAULT_WIDTH": "1024",
            "PROMPT_GEN_DEFAULT_HEIGHT": "768",
            "PROMPT_GEN_SSE_QUEUE_SIZE": "200",
            "PROMPT_GEN_IMAGE_SEED": "42",
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings.from_env()
            assert settings.lm_studio.base_url == "http://test:5000/v1"
            assert settings.image_generation.default_width == 1024
            assert settings.image_generation.default_height == 768
            assert settings.server.sse_queue_size == 200
            assert settings.image_generation.seed == 42

    def test_immutable_config(self):
        """Test that config dataclasses are immutable."""
        config = LMStudioConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.base_url = "http://changed"

    def test_invalid_env_vars(self):
        """Test that invalid environment variables fall back to defaults."""
        env_vars = {
            "PROMPT_GEN_DEFAULT_WIDTH": "not-an-integer",
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings.from_env()
            assert settings.image_generation.default_width == 864

    def test_invalid_server_timeouts(self):
        """Test that invalid server timeouts raise ValueError."""
        with pytest.raises(ValueError, match="sse_timeout must be positive"):
            ServerConfig(sse_timeout=0)
        with pytest.raises(ValueError, match="sse_timeout must be positive"):
            ServerConfig(sse_timeout=-1)
        with pytest.raises(ValueError, match="worker_timeout must be positive"):
            ServerConfig(worker_timeout=0)
        with pytest.raises(ValueError, match="worker_timeout must be positive"):
            ServerConfig(worker_timeout=-1)

    def test_invalid_lm_studio_timeouts(self):
        """Test that invalid LM Studio timeouts raise ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            LMStudioConfig(timeout=0)
        with pytest.raises(ValueError, match="timeout must be positive"):
            LMStudioConfig(timeout=-1)

    def test_invalid_image_dimensions(self):
        """Test that invalid image dimensions raise ValueError."""
        with pytest.raises(ValueError, match="default_width must be positive"):
            ImageGenerationConfig(default_width=0)
        with pytest.raises(ValueError, match="default_height must be positive"):
            ImageGenerationConfig(default_height=-1)
    def test_invalid_enhancement_config(self):
        """Test that invalid enhancement settings raise ValueError."""
        from config import EnhancementConfig

        with pytest.raises(ValueError, match="default_softness must be between 0 and 1"):
            EnhancementConfig(default_softness=-0.5)
        with pytest.raises(ValueError, match="default_softness must be between 0 and 1"):
            EnhancementConfig(default_softness=2.0)
        with pytest.raises(ValueError, match="default_scale must be at least 1"):
            EnhancementConfig(default_scale=0)
        with pytest.raises(ValueError, match="default_scale must be at least 1"):
            EnhancementConfig(default_scale=-3)

    def test_path_properties(self):
        """Verify path properties are Path objects and correct."""
        assert isinstance(paths.root_dir, Path)
        assert isinstance(paths.generated_dir, Path)
        assert isinstance(paths.grammars_dir, Path)
        assert isinstance(paths.prompts_dir, Path)
        assert isinstance(paths.saved_dir, Path)
        assert isinstance(paths.queue_path, Path)
        assert isinstance(paths.templates_dir, Path)

        # Verify path relationships
        assert paths.generated_dir == paths.root_dir / "generated"
        assert paths.grammars_dir == paths.generated_dir / "grammars"
        assert paths.prompts_dir == paths.generated_dir / "prompts"
        assert paths.saved_dir == paths.generated_dir / "saved"
        assert paths.queue_path == paths.generated_dir / "queue.json"
        assert paths.templates_dir == paths.root_dir / "templates"
