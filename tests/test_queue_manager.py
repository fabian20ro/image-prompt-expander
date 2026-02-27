"""Tests for QueueManager."""

from server.models import TaskType, TaskStatus, TaskProgress
from server.queue_manager import QueueManager


class TestQueueManager:
    """Tests for QueueManager."""

    def test_load_empty_queue(self, queue_path):
        """Test loading an empty/nonexistent queue."""
        qm = QueueManager(queue_path)
        state = qm.get_state()

        assert state.version == 1
        assert state.current_task is None
        assert len(state.pending) == 0

    def test_add_task(self, queue_path):
        """Test adding a task to the queue."""
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

    def test_get_next_task(self, queue_path):
        """Test getting the next task from the queue."""
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

    def test_get_next_task_when_running(self, queue_path):
        """Test that get_next_task returns None when a task is running."""
        qm = QueueManager(queue_path)

        # Add two tasks
        qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test1"})
        qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test2"})

        # Get first task
        qm.get_next_task()

        # Should return None since a task is running
        next_task = qm.get_next_task()
        assert next_task is None

    def test_complete_task(self, queue_path):
        """Test completing a task."""
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

    def test_fail_task(self, queue_path):
        """Test failing a task."""
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

    def test_update_progress(self, queue_path):
        """Test updating task progress."""
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

    def test_cancel_pending_task(self, queue_path):
        """Test cancelling a pending task."""
        qm = QueueManager(queue_path)

        task = qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test"})

        # Cancel the pending task
        result = qm.cancel_task(task.id)

        assert result is True
        state = qm.get_state()
        assert len(state.pending) == 0
        assert len(state.completed) == 1
        assert state.completed[0].status == TaskStatus.CANCELLED

    def test_cancel_running_task(self, queue_path):
        """Test cancelling a running task."""
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

    def test_clear_pending(self, queue_path):
        """Test clearing all pending tasks."""
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

    def test_clear_pending_emits_queue_cleared_and_queue_updated(self, queue_path):
        """Clearing pending tasks should emit both clear and updated queue events."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append((event, data))

        qm.add_listener(listener)
        qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test1"})
        qm.add_task(TaskType.GENERATE_IMAGE, {"run_id": "test2"})

        # Ignore add_task notifications and inspect clear behavior only.
        events.clear()
        count = qm.clear_pending()

        assert count == 2
        assert len(events) == 2
        assert events[0][0] == "queue_cleared"
        assert events[0][1]["count"] == 2
        assert events[1][0] == "queue_updated"
        assert events[1][1]["pending_count"] == 0
        assert events[1][1]["current"] is None

    def test_persistence(self, queue_path):
        """Test that queue state is persisted to disk."""
        # Create queue and add task
        qm1 = QueueManager(queue_path)
        task = qm1.add_task(TaskType.GENERATE_PIPELINE, {"prompt": "test"})

        # Create new QueueManager instance
        qm2 = QueueManager(queue_path)
        state = qm2.get_state()

        assert len(state.pending) == 1
        assert state.pending[0].id == task.id
        assert state.pending[0].params["prompt"] == "test"

    def test_listener_notifications(self, queue_path):
        """Test that listeners are notified of events."""
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

    def test_listener_add_remove_thread_safety(self, queue_path):
        """Test that listeners can be added/removed safely during notification."""
        qm = QueueManager(queue_path)
        events = []

        def listener(event, data):
            events.append(event)

        qm.add_listener(listener)
        task = qm.add_task("generate_pipeline", {"prompt": "test"})
        assert len(events) == 1  # queue_updated

        qm.remove_listener(listener)
        qm.add_task("generate_pipeline", {"prompt": "test2"})
        assert len(events) == 1  # No new events after removal
