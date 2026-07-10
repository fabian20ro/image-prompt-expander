"""Tests for interactive gallery generation."""

import json

from conftest import create_run_files
from gallery import generate_gallery_for_directory, update_gallery


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
        assert "btn-undo-grammar" in content
        assert "btn-redo-grammar" in content
        assert "grammar-history-list" in content
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

    def test_gallery_uses_persisted_layout_and_settings(self, temp_dir):
        """Gallery form defaults should come from persisted metadata, not hardcoded literals."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 3,
            "display_title": "Imported grammar run",
            "image_generation": {
                "model": "ernie-image-turbo",
                "width": 1024,
                "height": 768,
                "steps": 12,
                "seed": 7,
                "enhance": True,
                "enhance_softness": 0.3,
            },
            "gallery_layout": {
                "images_per_prompt": 3,
                "max_prompts": 2,
            },
        }
        create_run_files(run_dir, num_prompts=3, metadata=metadata)

        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        content = gallery_path.read_text()

        assert 'value="3"' in content
        assert 'value="2"' in content
        assert 'value="1024"' in content
        assert 'value="768"' in content
        assert 'id="img-steps"' not in content
        assert 'id="img-model"' not in content
        assert 'value="7"' in content
        assert "Imported grammar run" in content
        assert content.count('data-prompt-idx="2"') == 0

    def test_gallery_preserves_prompt_only_layout(self, temp_dir):
        """A persisted zero-images layout should render prompt-only cards."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 2,
            "user_prompt": "prompt-only run",
            "gallery_layout": {
                "images_per_prompt": 0,
                "max_prompts": 2,
            },
        }
        create_run_files(run_dir, num_prompts=2, metadata=metadata)

        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        content = gallery_path.read_text()

        assert 'class="card prompt-only"' in content
        assert 'Images/Prompt (0 = prompt-only layout)' in content
        assert 'id="img-images-per-prompt" name="images_per_prompt" value="0" min="0"' in content
        assert "prompt-only run" in content
        assert "Pending..." not in content

    def test_update_gallery(self, temp_dir):
        """Test that update_gallery correctly replaces placeholders and updates status."""
        run_dir = temp_dir
        prefix = "test_update"
        gallery_path = run_dir / f"{prefix}_gallery.html"
        image_path = run_dir / f"{prefix}_0_0.png"

        # Create a dummy gallery with a placeholder
        gallery_path.write_text(f'''<div class="card" data-image="{prefix}_0_0.png" data-prompt-idx="0" data-image-idx="0">
            <div class="placeholder">Pending...</div>
          </div>
          <p class="status">Generated: 0 / 1 images</p>''')

        # Create a dummy image
        image_path.write_text("image data")

        from gallery import update_gallery
        update_gallery(gallery_path, image_path, "test prompt", 0, 1)

        content = gallery_path.read_text()
        assert f'<a href="{image_path.name}" target="_blank">' in content
        assert '<img src="' in content
        assert '<p class="status">Generated: 0 / 1 images</p>' in content

    def test_update_gallery_preserves_sibling_action_buttons(self, temp_dir):
        """update_gallery should replace the placeholder while keeping sibling markup intact."""
        run_dir = temp_dir
        prefix = "test_siblings"
        gallery_path = run_dir / f"{prefix}_gallery.html"
        image_path = run_dir / f"{prefix}_0_0.png"

        # Create a realistic card with a pending placeholder AND action buttons (as in interactive mode)
        gallery_path.write_text(f'''<div class="card" data-image="{prefix}_0_0.png" data-prompt-idx="0" data-image-idx="0">
          <div class="placeholder">Pending...</div>
          <div class="prompt">Initial prompt text</div><div class="card-actions">
        <button class="btn-small btn-primary" onclick="generateImage(this, 0, 0)">Generate</button>
        <button class="btn-small btn-secondary" onclick="enhanceImage(this, 0, 0)">Enhance</button>
      </div></div>''')

        image_path.write_text("new image data")

        from gallery import update_gallery
        update_gallery(gallery_path, image_path, "Updated prompt", 1, 2)

        content = gallery_path.read_text()
        # Placeholder should be replaced with actual image link (alt is the escaped prompt)
        assert '<img src="' in content
        assert f'alt="Updated prompt"' in content
        # Action buttons must remain untouched after the replacement
        assert "btn-primary" in content
        assert "btn-secondary" in content
        assert "generateImage(this, 0, 0)" in content

    def test_update_gallery_skips_missing_gallery(self, temp_dir):
        """update_gallery must be a no-op when the gallery HTML is absent."""
        from pathlib import Path

        run_dir = temp_dir
        missing_path = run_dir / "does_not_exist_gallery.html"
        image_path = run_dir / "test_0_0.png"

        # Call update_gallery on a non-existent gallery — it should not raise.
        update_gallery(missing_path, image_path, "prompt", 1, 1)
        assert not missing_path.exists()
