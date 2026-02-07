"""Tests for interactive gallery index generation."""

import json

from gallery_index import generate_master_index


class TestGalleryIndexInteractive:
    """Tests for interactive gallery index generation."""

    def test_index_interactive_mode(self, temp_dir):
        """Test that interactive index includes generation form."""
        (temp_dir / "prompts").mkdir()

        # Generate interactive index
        index_path = generate_master_index(temp_dir, interactive=True)

        assert index_path.exists()
        content = index_path.read_text()

        # Check for interactive elements
        assert "generate-form" in content
        assert "queue-status" in content
        assert "btn-kill" in content
        assert "btn-clear" in content
        assert "/api/generate" in content
        assert "EventSource" in content

    def test_index_non_interactive_mode(self, temp_dir):
        """Test that non-interactive index doesn't include form."""
        (temp_dir / "prompts").mkdir()

        # Generate non-interactive index
        index_path = generate_master_index(temp_dir, interactive=False)

        content = index_path.read_text()

        # Check that form is NOT present
        assert "generate-form" not in content
        assert "queue-status" not in content

    def test_index_interactive_uses_toasts_modal_and_no_blocking_dialogs(self, temp_dir):
        """Interactive index should use shared toast/modal helpers instead of alert/confirm."""
        (temp_dir / "prompts").mkdir()

        index_path = generate_master_index(temp_dir, interactive=True)
        content = index_path.read_text()

        assert "toast-region" in content
        assert "confirm-modal" in content
        assert "showToast(" in content
        assert "confirmAction(" in content
        assert "withButtonBusy(" in content
        assert "queue_cleared" in content

        # Guard against regressions to blocking browser dialogs.
        assert "alert(" not in content
        assert "confirm(" not in content

    def test_index_with_archived_runs(self, temp_dir):
        """Test that index shows archived runs separately."""
        # Create active run
        active_run = temp_dir / "prompts" / "20240101_120000_abc123"
        active_run.mkdir(parents=True)
        (active_run / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
            "user_prompt": "active prompt",
        }))
        (active_run / "test_gallery.html").write_text("<html></html>")

        # Create archived run
        archived_run = temp_dir / "saved" / "20240101_100000_def456_20240101_130000"
        archived_run.mkdir(parents=True)
        (archived_run / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
            "user_prompt": "archived prompt",
            "backup_info": {
                "is_backup": True,
                "backup_reason": "pre_regenerate",
                "source_run_id": "20240101_100000_def456",
            },
        }))
        (archived_run / "test_gallery.html").write_text("<html></html>")

        # Generate index
        index_path = generate_master_index(temp_dir, interactive=True)
        content = index_path.read_text()

        # Check for both runs
        assert "active prompt" in content
        assert "archived prompt" in content
        assert "Saved Archives" in content
        assert "archive-badge" in content
        assert "Pre-Regen" in content  # Badge for pre_regenerate
