"""Tests for interactive gallery generation."""

import json

from conftest import create_run_files
from gallery import generate_gallery_for_directory


class TestGalleryInteractive:
    """Tests for interactive gallery generation."""

    def test_gallery_interactive_mode(self, temp_dir, sample_grammar):
        """Test that interactive gallery includes editor and buttons."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 2,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, metadata=metadata, grammar=sample_grammar)

        # Generate interactive gallery
        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)

        assert gallery_path.exists()
        content = gallery_path.read_text()

        # Check for interactive elements
        assert "grammar-editor" in content
        assert "btn-save-grammar" in content
        assert "btn-regenerate" in content
        assert "btn-generate-all" in content
        assert "btn-enhance-all" in content
        assert "generateImage" in content
        assert "enhanceImage" in content

    def test_gallery_interactive_has_nav_and_archive(self, temp_dir):
        """Test that interactive gallery includes nav header and archive button."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 1,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, num_prompts=1, metadata=metadata)

        # Generate interactive gallery
        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        content = gallery_path.read_text()

        # Check for nav header
        assert "nav-header" in content
        assert "Back to Index" in content
        assert "/index" in content

        # Check for archive button
        assert "btn-archive" in content
        assert "Save to Archive" in content

    def test_gallery_non_interactive_mode(self, temp_dir):
        """Test that non-interactive gallery doesn't include interactive elements."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 1,
            "user_prompt": "test",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, num_prompts=1, metadata=metadata)

        # Generate non-interactive gallery
        gallery_path = generate_gallery_for_directory(run_dir, interactive=False)

        content = gallery_path.read_text()

        # Check that interactive elements are NOT present
        assert "grammar-editor" not in content
        assert "btn-generate-all" not in content
        assert "generateImage(" not in content

    def test_gallery_interactive_uses_toasts_modal_and_busy_button_handlers(self, temp_dir):
        """Interactive gallery should use non-blocking notifications and button context."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 1,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, num_prompts=1, metadata=metadata)

        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        content = gallery_path.read_text()

        assert "toast-region" in content
        assert "confirm-modal" in content
        assert "showToast(" in content
        assert "confirmAction(" in content
        assert "withButtonBusy(" in content
        assert "queue_cleared" in content
        assert "generateImage(this," in content
        assert "enhanceImage(this," in content

        # Guard against regressions back to blocking browser dialogs.
        assert "alert(" not in content
        assert "confirm(" not in content
