"""Tests for API routes."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import models first (no circular imports)
from server.models import TaskStatus, Task, TaskType, QueueState


@pytest.fixture
def mock_queue_manager():
    """Create a mock queue manager."""
    mock_qm = MagicMock()
    mock_qm.get_state.return_value = QueueState()
    mock_qm.add_task.return_value = Task(
        id="test-task-123",
        type=TaskType.GENERATE_PIPELINE,
        status=TaskStatus.PENDING,
    )
    return mock_qm


@pytest.fixture
def mock_worker():
    """Create a mock worker."""
    mock_w = MagicMock()
    mock_w.kill_current = AsyncMock(return_value=True)
    return mock_w


@pytest.fixture
def client(temp_dir, mock_queue_manager, mock_worker):
    """Create test client with mocked dependencies."""
    # Create minimal directory structure
    prompts_dir = temp_dir / "prompts"
    saved_dir = temp_dir / "saved"
    prompts_dir.mkdir()
    saved_dir.mkdir()

    # Create fresh app to avoid circular import issues
    app = FastAPI()

    # Patch before importing routes
    with patch("server.app.get_queue_manager", return_value=mock_queue_manager):
        with patch("server.app.get_worker", return_value=mock_worker):
            # Import router inside the patch context
            import server.routes as routes_module
            from services.gallery_service import GalleryService

            # Patch the functions in routes module
            routes_module.get_queue_manager = lambda: mock_queue_manager
            routes_module.get_worker = lambda: mock_worker

            # Patch paths
            routes_module.paths = MagicMock()
            routes_module.paths.prompts_dir = prompts_dir
            routes_module.paths.saved_dir = saved_dir
            routes_module.paths.generated_dir = temp_dir

            # Reset the global gallery service to use temp_dir paths
            routes_module._gallery_service = GalleryService(prompts_dir, saved_dir)

            app.include_router(routes_module.router)

            with TestClient(app) as client:
                yield client


class TestGenerateEndpoint:
    """Tests for /api/generate endpoint."""

    def test_generate_valid_request(self, client, mock_queue_manager):
        """Test generating with valid request."""
        response = client.post("/api/generate", json={
            "prompt": "a dragon flying",
            "count": 10,
        })

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "message" in data

    def test_generate_empty_prompt_rejected(self, client):
        """Test that empty prompt is rejected."""
        response = client.post("/api/generate", json={
            "prompt": "",
        })

        assert response.status_code == 422  # Validation error

    def test_generate_whitespace_prompt_rejected(self, client):
        """Test that whitespace-only prompt is rejected."""
        response = client.post("/api/generate", json={
            "prompt": "   ",
        })

        assert response.status_code == 400
        assert "required" in response.json()["detail"].lower()

    def test_generate_count_bounds(self, client):
        """Test count validation bounds."""
        # Too low
        response = client.post("/api/generate", json={
            "prompt": "test",
            "count": 0,
        })
        assert response.status_code == 422

        # Too high
        response = client.post("/api/generate", json={
            "prompt": "test",
            "count": 20000,
        })
        assert response.status_code == 422


class TestGalleryEndpoint:
    """Tests for /gallery/{run_id} endpoint."""

    def test_gallery_not_found(self, client):
        """Test 404 for missing gallery."""
        response = client.get("/gallery/nonexistent_gallery")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_gallery_exists(self, client, temp_dir):
        """Test serving existing gallery."""
        # Create a gallery
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "count": 1,
            "user_prompt": "test",
            "image_generation": {"images_per_prompt": 1},
        }))
        (run_dir / "test_grammar.json").write_text(json.dumps({"origin": ["test"]}))
        (run_dir / "test_0.txt").write_text("Test prompt")

        response = client.get("/gallery/20240101_120000_abc123")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestGrammarUpdateEndpoint:
    """Tests for PUT /api/gallery/{run_id}/grammar."""

    def test_update_grammar_invalid_json(self, client, temp_dir):
        """Test that invalid JSON is rejected."""
        # Create a gallery
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
        }))
        (run_dir / "test_grammar.json").write_text(json.dumps({"origin": ["test"]}))

        response = client.put("/api/gallery/20240101_120000_abc123/grammar", json={
            "grammar": "not valid json {",
        })

        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]

    def test_update_grammar_gallery_not_found(self, client):
        """Test 404 for updating grammar of missing gallery."""
        response = client.put("/api/gallery/nonexistent/grammar", json={
            "grammar": '{"origin": ["test"]}',
        })

        assert response.status_code == 404

    def test_update_grammar_success(self, client, temp_dir):
        """Test successful grammar update."""
        # Create a gallery
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
        }))
        (run_dir / "test_grammar.json").write_text(json.dumps({"origin": ["old"]}))

        new_grammar = '{"origin": ["new"]}'
        response = client.put("/api/gallery/20240101_120000_abc123/grammar", json={
            "grammar": new_grammar,
        })

        assert response.status_code == 200
        assert response.json()["message"] == "Grammar updated"

        # Verify file was updated
        updated = (run_dir / "test_grammar.json").read_text()
        assert updated == new_grammar


class TestDeleteGalleryEndpoint:
    """Tests for DELETE /api/gallery/{run_id}."""

    def test_delete_gallery_not_found(self, client):
        """Test 404 for deleting missing gallery."""
        response = client.delete("/api/gallery/nonexistent")
        assert response.status_code == 404

    def test_delete_archive_protected(self, client, temp_dir):
        """Test that archives cannot be deleted (returns 400)."""
        # Create an archive (backup)
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "backup_info": {
                "is_backup": True,
                "source_run_id": "original",
            },
        }))

        response = client.delete("/api/gallery/20240101_120000_abc123")
        assert response.status_code == 400
        assert "archived" in response.json()["detail"].lower()

    def test_delete_active_gallery_queued(self, client, temp_dir, mock_queue_manager):
        """Test that delete queues a task for active gallery."""
        # Create an active gallery (not a backup)
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
        }))

        response = client.delete("/api/gallery/20240101_120000_abc123")
        assert response.status_code == 200
        assert "queued" in response.json()["message"].lower()


class TestSavedFileEndpoint:
    """Tests for /saved/{filename} endpoint."""

    def test_saved_file_not_found(self, client):
        """Test 404 for missing file."""
        response = client.get("/saved/nonexistent.png")
        assert response.status_code == 404

    def test_saved_file_non_png_rejected(self, client, temp_dir):
        """Test that non-PNG files are rejected."""
        saved_dir = temp_dir / "saved"
        (saved_dir / "test.txt").write_text("not an image")

        response = client.get("/saved/test.txt")
        assert response.status_code == 400
        assert "PNG" in response.json()["detail"]

    def test_saved_file_success(self, client, temp_dir):
        """Test serving a valid saved PNG."""
        saved_dir = temp_dir / "saved"
        (saved_dir / "test_20240101_0_0.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake png data")

        response = client.get("/saved/test_20240101_0_0.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


class TestArchiveGalleryEndpoint:
    """Tests for POST /api/gallery/{run_id}/archive."""

    def test_archive_gallery_not_found(self, client):
        """Test 404 for archiving missing gallery."""
        response = client.post("/api/gallery/nonexistent/archive")
        assert response.status_code == 404

    def test_archive_backup_rejected(self, client, temp_dir):
        """Test that archiving a backup is rejected."""
        prompts_dir = temp_dir / "prompts"
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        (run_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "backup_info": {"is_backup": True},
        }))

        response = client.post("/api/gallery/20240101_120000_abc123/archive")
        assert response.status_code == 400
        assert "backup" in response.json()["detail"].lower()


class TestQueueEndpoints:
    """Tests for queue management endpoints."""

    def test_clear_queue(self, client, mock_queue_manager):
        """Test clearing the queue."""
        mock_queue_manager.clear_pending.return_value = 5

        response = client.post("/api/queue/clear")
        assert response.status_code == 200
        assert "5" in response.json()["message"]

    def test_get_status(self, client, mock_queue_manager):
        """Test getting queue status."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "queue_length" in data
        assert "pending_count" in data


class TestKillWorkerEndpoint:
    """Tests for /api/worker/kill endpoint."""

    def test_kill_worker_success(self, client, mock_worker):
        """Test killing worker when task is running."""
        mock_worker.kill_current = AsyncMock(return_value=True)

        response = client.post("/api/worker/kill")
        assert response.status_code == 200
        assert "Killed" in response.json()["message"]

    def test_kill_worker_no_task(self, client, mock_worker):
        """Test killing worker when no task is running."""
        mock_worker.kill_current = AsyncMock(return_value=False)

        response = client.post("/api/worker/kill")
        assert response.status_code == 200
        assert "No task" in response.json()["message"]
