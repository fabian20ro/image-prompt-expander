"""Tests for the web UI server components."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from server.models import (
    TaskType,
    TaskStatus,
    Task,
    QueueState,
    TaskProgress,
    GenerateRequest,
)
from server.queue_manager import QueueManager


class TestModels:
    """Tests for Pydantic models."""

    def test_task_type_enum(self):
        """Test TaskType enum values."""
        assert TaskType.GENERATE_PIPELINE.value == "generate_pipeline"
        assert TaskType.REGENERATE_PROMPTS.value == "regenerate_prompts"
        assert TaskType.GENERATE_IMAGE.value == "generate_image"
        assert TaskType.ENHANCE_IMAGE.value == "enhance_image"
        assert TaskType.GENERATE_ALL_IMAGES.value == "generate_all_images"
        assert TaskType.ENHANCE_ALL_IMAGES.value == "enhance_all_images"

    def test_task_status_enum(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_task_creation(self):
        """Test Task model creation with defaults."""
        task = Task(
            id="test-id",
            type=TaskType.GENERATE_PIPELINE,
        )
        assert task.id == "test-id"
        assert task.type == TaskType.GENERATE_PIPELINE
        assert task.status == TaskStatus.PENDING
        assert task.pid is None
        assert task.params == {}

    def test_task_progress(self):
        """Test TaskProgress model."""
        progress = TaskProgress(
            stage="generating_images",
            current=5,
            total=10,
            message="Generating image_5.png",
        )
        assert progress.stage == "generating_images"
        assert progress.current == 5
        assert progress.total == 10
        assert progress.message == "Generating image_5.png"

    def test_queue_state_empty(self):
        """Test empty QueueState."""
        state = QueueState()
        assert state.version == 1
        assert state.current_task is None
        assert state.pending == []
        assert state.completed == []

    def test_generate_request_defaults(self):
        """Test GenerateRequest with default values."""
        req = GenerateRequest(prompt="a dragon")
        assert req.prompt == "a dragon"
        assert req.count == 50
        assert req.prefix == "image"
        assert req.model == "z-image-turbo"
        assert req.temperature == 0.7
        assert req.no_cache is False
        assert req.generate_images is False
        assert req.width == 864
        assert req.height == 1152
        assert req.quantize == 8
        assert req.tiled_vae is True
        assert req.enhance is False

    def test_generate_request_custom(self):
        """Test GenerateRequest with custom values."""
        req = GenerateRequest(
            prompt="a cat",
            count=10,
            prefix="cat",
            model="flux2-klein-4b",
            generate_images=True,
            enhance=True,
            enhance_softness=0.3,
        )
        assert req.prompt == "a cat"
        assert req.count == 10
        assert req.prefix == "cat"
        assert req.model == "flux2-klein-4b"
        assert req.generate_images is True
        assert req.enhance is True
        assert req.enhance_softness == 0.3


class TestQueueManager:
    """Tests for QueueManager."""

    def test_load_empty_queue(self):
        """Test loading an empty/nonexistent queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)
            state = qm.get_state()

            assert state.version == 1
            assert state.current_task is None
            assert len(state.pending) == 0

    def test_add_task(self):
        """Test adding a task to the queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)

            task = qm.add_task(
                TaskType.GENERATE_PIPELINE,
                {"prompt": "test prompt", "count": 10},
            )

            assert task.type == TaskType.GENERATE_PIPELINE
            assert task.status == TaskStatus.PENDING
            assert task.params["prompt"] == "test prompt"
            assert task.params["count"] == 10

            # Verify state was saved
            state = qm.get_state()
            assert len(state.pending) == 1
            assert state.pending[0].id == task.id

    def test_get_next_task(self):
        """Test getting the next task from the queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)

            # Add a task
            task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})

            # Get the task
            next_task = qm.get_next_task()

            assert next_task is not None
            assert next_task.id == task.id
            assert next_task.status == TaskStatus.RUNNING

            # Verify queue state
            state = qm.get_state()
            assert len(state.pending) == 0
            assert state.current_task is not None
            assert state.current_task.id == task.id

    def test_get_next_task_when_running(self):
        """Test that get_next_task returns None when a task is running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)

            # Add two tasks
            qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test1"})
            qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test2"})

            # Get first task
            qm.get_next_task()

            # Should return None since a task is running
            next_task = qm.get_next_task()
            assert next_task is None

    def test_complete_task(self):
        """Test completing a task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)

            task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
            qm.get_next_task()

            # Complete the task
            qm.complete_task(task.id, {"generated": 5})

            state = qm.get_state()
            assert state.current_task is None
            assert len(state.completed) == 1
            assert state.completed[0].status == TaskStatus.COMPLETED
            assert state.completed[0].result == {"generated": 5}

    def test_fail_task(self):
        """Test failing a task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)

            task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
            qm.get_next_task()

            # Fail the task
            qm.fail_task(task.id, "Something went wrong")

            state = qm.get_state()
            assert state.current_task is None
            assert len(state.completed) == 1
            assert state.completed[0].status == TaskStatus.FAILED
            assert state.completed[0].error == "Something went wrong"

    def test_update_progress(self):
        """Test updating task progress."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)

            task = qm.add_task(TaskType.GENERATE_PIPELINE, {"prompt": "test"})
            qm.get_next_task()

            # Update progress
            progress = TaskProgress(
                stage="generating_images",
                current=5,
                total=10,
                message="Working...",
            )
            qm.update_progress(task.id, progress)

            state = qm.get_state()
            assert state.current_task.progress.stage == "generating_images"
            assert state.current_task.progress.current == 5
            assert state.current_task.progress.total == 10

    def test_cancel_pending_task(self):
        """Test cancelling a pending task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)

            task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})

            # Cancel the pending task
            result = qm.cancel_task(task.id)

            assert result is True
            state = qm.get_state()
            assert len(state.pending) == 0
            assert len(state.completed) == 1
            assert state.completed[0].status == TaskStatus.CANCELLED

    def test_cancel_running_task(self):
        """Test cancelling a running task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)

            task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})
            qm.get_next_task()

            # Cancel the running task
            result = qm.cancel_task(task.id)

            assert result is True
            state = qm.get_state()
            assert state.current_task is None
            assert len(state.completed) == 1
            assert state.completed[0].status == TaskStatus.CANCELLED

    def test_clear_pending(self):
        """Test clearing all pending tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)

            # Add multiple tasks
            qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test1"})
            qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test2"})
            qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test3"})

            # Clear pending
            count = qm.clear_pending()

            assert count == 3
            state = qm.get_state()
            assert len(state.pending) == 0

    def test_persistence(self):
        """Test that queue state is persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"

            # Create queue and add task
            qm1 = QueueManager(queue_path)
            task = qm1.add_task(TaskType.GENERATE_PIPELINE, {"prompt": "test"})

            # Create new QueueManager instance
            qm2 = QueueManager(queue_path)
            state = qm2.get_state()

            assert len(state.pending) == 1
            assert state.pending[0].id == task.id
            assert state.pending[0].params["prompt"] == "test"

    def test_listener_notifications(self):
        """Test that listeners are notified of events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.json"
            qm = QueueManager(queue_path)

            events = []

            def listener(event, data):
                events.append((event, data))

            qm.add_listener(listener)

            # Add a task
            task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})

            assert len(events) == 1
            assert events[0][0] == "queue_updated"

            # Get next task
            qm.get_next_task()

            assert len(events) == 2
            assert events[1][0] == "task_started"

            # Complete task
            qm.complete_task(task.id, {"result": "ok"})

            assert len(events) == 3
            assert events[2][0] == "task_completed"


class TestGalleryInteractive:
    """Tests for interactive gallery generation."""

    def test_gallery_interactive_mode(self):
        """Test that interactive gallery includes editor and buttons."""
        from gallery import generate_gallery_for_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock data
            (tmpdir / "test_0.txt").write_text("Prompt 0")
            (tmpdir / "test_1.txt").write_text("Prompt 1")
            (tmpdir / "test_metadata.json").write_text(json.dumps({
                "prefix": "test",
                "count": 2,
                "user_prompt": "test prompt",
                "image_generation": {"images_per_prompt": 1},
            }))
            (tmpdir / "test_grammar.json").write_text(json.dumps({
                "origin": ["#subject# in #setting#"],
                "subject": ["a cat", "a dog"],
                "setting": ["a garden", "a forest"],
            }))

            # Generate interactive gallery
            gallery_path = generate_gallery_for_directory(tmpdir, interactive=True)

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

    def test_gallery_non_interactive_mode(self):
        """Test that non-interactive gallery doesn't include interactive elements."""
        from gallery import generate_gallery_for_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock data
            (tmpdir / "test_0.txt").write_text("Prompt 0")
            (tmpdir / "test_metadata.json").write_text(json.dumps({
                "prefix": "test",
                "count": 1,
                "user_prompt": "test",
                "image_generation": {"images_per_prompt": 1},
            }))
            (tmpdir / "test_grammar.json").write_text(json.dumps({"origin": ["test"]}))

            # Generate non-interactive gallery
            gallery_path = generate_gallery_for_directory(tmpdir, interactive=False)

            content = gallery_path.read_text()

            # Check that interactive elements are NOT present
            assert "grammar-editor" not in content
            assert "btn-generate-all" not in content
            assert "generateImage(" not in content


class TestGalleryIndexInteractive:
    """Tests for interactive gallery index generation."""

    def test_index_interactive_mode(self):
        """Test that interactive index includes generation form."""
        from gallery_index import generate_master_index

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "prompts").mkdir()

            # Generate interactive index
            index_path = generate_master_index(tmpdir, interactive=True)

            assert index_path.exists()
            content = index_path.read_text()

            # Check for interactive elements
            assert "generate-form" in content
            assert "queue-status" in content
            assert "btn-kill" in content
            assert "btn-clear" in content
            assert "/api/generate" in content
            assert "EventSource" in content

    def test_index_non_interactive_mode(self):
        """Test that non-interactive index doesn't include form."""
        from gallery_index import generate_master_index

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "prompts").mkdir()

            # Generate non-interactive index
            index_path = generate_master_index(tmpdir, interactive=False)

            content = index_path.read_text()

            # Check that form is NOT present
            assert "generate-form" not in content
            assert "queue-status" not in content
