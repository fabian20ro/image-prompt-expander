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


class TestConfig:
    """Tests for centralized configuration."""

    def test_default_settings(self):
        """Test that default settings are loaded correctly."""
        from config import Settings, LMStudioConfig, ImageGenerationConfig

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
        import os
        from config import Settings

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
        from config import LMStudioConfig

        config = LMStudioConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.base_url = "http://changed"


class TestUtils:
    """Tests for utility functions."""

    def test_load_run_metadata(self):
        """Test loading metadata from a run directory."""
        from utils import load_run_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "test_metadata.json").write_text(json.dumps({
                "prefix": "test",
                "count": 10,
                "user_prompt": "a dragon",
            }))

            metadata = load_run_metadata(tmpdir)
            assert metadata["prefix"] == "test"
            assert metadata["count"] == 10

    def test_load_run_metadata_not_found(self):
        """Test that ValueError is raised when no metadata found."""
        from utils import load_run_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No metadata file found"):
                load_run_metadata(Path(tmpdir))

    def test_get_prefix_from_metadata(self):
        """Test getting prefix from metadata."""
        from utils import get_prefix_from_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "cat_metadata.json").write_text(json.dumps({
                "prefix": "cat",
            }))

            prefix = get_prefix_from_metadata(tmpdir)
            assert prefix == "cat"

    def test_get_prefix_from_metadata_default(self):
        """Test that default prefix is returned when metadata missing."""
        from utils import get_prefix_from_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = get_prefix_from_metadata(Path(tmpdir))
            assert prefix == "image"

    def test_count_images_in_run(self):
        """Test counting images in a run directory."""
        from utils import count_images_in_run

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))
            (tmpdir / "test_0_0.png").write_text("fake image")
            (tmpdir / "test_0_1.png").write_text("fake image")
            (tmpdir / "test_1_0.png").write_text("fake image")

            count = count_images_in_run(tmpdir)
            assert count == 3

    def test_get_prompts_from_run(self):
        """Test loading prompts from a run directory."""
        from utils import get_prompts_from_run

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))
            (tmpdir / "test_0.txt").write_text("First prompt")
            (tmpdir / "test_1.txt").write_text("Second prompt")
            (tmpdir / "test_2.txt").write_text("Third prompt")

            prompts = get_prompts_from_run(tmpdir)
            assert len(prompts) == 3
            assert prompts[0] == "First prompt"
            assert prompts[2] == "Third prompt"


class TestInputValidation:
    """Tests for Pydantic model input validation."""

    def test_generate_request_prompt_required(self):
        """Test that prompt is required."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GenerateRequest()  # Missing required prompt

    def test_generate_request_prompt_not_empty(self):
        """Test that prompt cannot be empty."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="")

    def test_generate_request_count_bounds(self):
        """Test that count must be within bounds."""
        from pydantic import ValidationError

        # Too low
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", count=0)

        # Too high
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", count=10001)

        # Just right
        req = GenerateRequest(prompt="test", count=100)
        assert req.count == 100

    def test_generate_request_dimensions_bounds(self):
        """Test that width/height must be within bounds."""
        from pydantic import ValidationError

        # Too small
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", width=32)

        # Too large
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", height=5000)

        # Just right
        req = GenerateRequest(prompt="test", width=512, height=768)
        assert req.width == 512
        assert req.height == 768

    def test_generate_request_temperature_bounds(self):
        """Test that temperature must be within bounds."""
        from pydantic import ValidationError

        # Too low
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", temperature=-0.1)

        # Too high
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", temperature=2.5)

        # Just right
        req = GenerateRequest(prompt="test", temperature=1.5)
        assert req.temperature == 1.5

    def test_generate_request_prefix_pattern(self):
        """Test that prefix must match allowed pattern."""
        from pydantic import ValidationError

        # Valid prefixes
        GenerateRequest(prompt="test", prefix="image")
        GenerateRequest(prompt="test", prefix="my-prefix")
        GenerateRequest(prompt="test", prefix="prefix_123")

        # Invalid prefixes
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", prefix="prefix with spaces")

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="test", prefix="prefix.with.dots")

    def test_enhance_softness_bounds(self):
        """Test that softness must be within 0-1."""
        from pydantic import ValidationError
        from server.models import EnhanceImageRequest

        # Too low
        with pytest.raises(ValidationError):
            EnhanceImageRequest(softness=-0.1)

        # Too high
        with pytest.raises(ValidationError):
            EnhanceImageRequest(softness=1.5)

        # Just right
        req = EnhanceImageRequest(softness=0.7)
        assert req.softness == 0.7
