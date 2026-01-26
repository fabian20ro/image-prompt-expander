"""Tests for centralized configuration."""

import os
from pathlib import Path

import pytest

from config import Settings, LMStudioConfig, paths


class TestConfig:
    """Tests for centralized configuration."""

    def test_default_settings(self):
        """Test that default settings are loaded correctly."""
        settings = Settings()

        assert settings.lm_studio.base_url == "http://localhost:1234/v1"
        assert settings.lm_studio.api_key == "lm-studio"
        assert settings.image_generation.default_width == 864
        assert settings.image_generation.default_height == 1152
        assert settings.image_generation.default_model == "z-image-turbo"
        assert settings.server.sse_queue_size == 100
        assert settings.enhancement.default_softness == 0.5

    def test_settings_from_env(self):
        """Test that settings can be loaded from environment variables."""
        # Save original env vars
        original = os.environ.get("PROMPT_GEN_LM_STUDIO_URL")

        try:
            os.environ["PROMPT_GEN_LM_STUDIO_URL"] = "http://test:5000/v1"
            settings = Settings.from_env()
            assert settings.lm_studio.base_url == "http://test:5000/v1"
        finally:
            # Restore original
            if original is not None:
                os.environ["PROMPT_GEN_LM_STUDIO_URL"] = original
            else:
                os.environ.pop("PROMPT_GEN_LM_STUDIO_URL", None)

    def test_immutable_config(self):
        """Test that config dataclasses are immutable."""
        config = LMStudioConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.base_url = "http://changed"

    def test_path_config(self):
        """Test that path configuration provides correct paths."""
        # Verify paths are Path objects
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
