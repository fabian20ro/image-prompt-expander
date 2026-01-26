"""Tests for worker.py - background task processor."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

# Import after path setup in conftest
from server.worker import Worker
from server.models import TaskProgress, TaskType


class MockTask:
    """Mock task object for testing."""

    def __init__(self, task_id="test-task-id", task_type=TaskType.GENERATE_PIPELINE, params=None):
        self.id = task_id
        self.type = task_type
        self.params = params or {"prompt": "test"}


class MockQueueState:
    """Mock queue state for testing."""

    def __init__(self, current_task=None):
        self.current_task = current_task


@pytest.fixture
def mock_queue_manager():
    """Create a mock QueueManager."""
    manager = MagicMock()
    manager.get_next_task = MagicMock(return_value=None)
    manager.get_state = MagicMock(return_value=MockQueueState())
    manager.update_task_pid = MagicMock()
    manager.update_progress = MagicMock()
    manager.notify_image_ready = MagicMock()
    manager.fail_task = MagicMock()
    manager.complete_task = MagicMock()
    manager.cancel_task = MagicMock()
    manager.emit_log = MagicMock()
    return manager


@pytest.fixture
def worker(mock_queue_manager, temp_dir):
    """Create a Worker instance for testing."""
    return Worker(mock_queue_manager, temp_dir)


class TestWorkerInit:
    """Tests for Worker initialization."""

    def test_init_sets_attributes(self, mock_queue_manager, temp_dir):
        """Test that Worker initializes with correct attributes."""
        worker = Worker(mock_queue_manager, temp_dir)

        assert worker.queue_manager is mock_queue_manager
        assert worker.generated_dir == temp_dir
        assert worker._running is False
        assert worker._current_process is None

    def test_worker_script_path_exists(self, worker):
        """Test that worker script path is set correctly."""
        assert worker._worker_script.name == "worker_subprocess.py"


class TestWorkerStop:
    """Tests for Worker.stop() method."""

    def test_stop_sets_running_false(self, worker):
        """Test that stop() sets _running to False."""
        worker._running = True
        worker.stop()
        assert worker._running is False

    def test_stop_terminates_process(self, worker):
        """Test that stop() terminates current process."""
        mock_process = MagicMock()
        worker._current_process = mock_process

        worker.stop()

        mock_process.terminate.assert_called_once()

    def test_stop_handles_no_process(self, worker):
        """Test that stop() handles case with no running process."""
        worker._running = True
        worker._current_process = None

        # Should not raise
        worker.stop()
        assert worker._running is False


class TestWorkerKillCurrent:
    """Tests for Worker.kill_current() method."""

    @pytest.mark.asyncio
    async def test_kill_current_no_process(self, worker):
        """Test kill_current() when no process is running."""
        worker._current_process = None
        result = await worker.kill_current()
        assert result is False

    @pytest.mark.asyncio
    async def test_kill_current_terminates_process(self, worker):
        """Test kill_current() terminates the current process."""
        # Create mock process with async wait
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()

        # Make wait() return a completed future
        async def mock_wait():
            return 0

        mock_process.wait = mock_wait

        worker._current_process = mock_process
        worker.queue_manager.get_state.return_value = MockQueueState(
            current_task=MockTask()
        )

        result = await worker.kill_current()

        assert result is True
        mock_process.terminate.assert_called_once()
        worker.queue_manager.cancel_task.assert_called()
        assert worker._current_process is None


class TestWorkerReadStderr:
    """Tests for Worker._read_stderr() method."""

    @pytest.mark.asyncio
    async def test_read_stderr_streams_lines(self, worker):
        """Test that stderr lines are streamed to queue manager."""
        lines = [b"Line 1\n", b"Line 2\n", b""]
        line_idx = 0

        class MockStderr:
            async def readline(self):
                nonlocal line_idx
                if line_idx < len(lines):
                    result = lines[line_idx]
                    line_idx += 1
                    return result
                return b""

        mock_process = MagicMock()
        mock_process.stderr = MockStderr()

        result = await worker._read_stderr(mock_process, "test-task-id")

        assert result == ["Line 1", "Line 2"]
        assert worker.queue_manager.emit_log.call_count == 2


class TestWorkerExecuteTask:
    """Tests for Worker._execute_task() method."""

    def _create_mock_process(self, stdout_lines, stderr_lines, return_code=0):
        """Create a mock process with given stdout/stderr lines."""
        stdout_idx = 0
        stderr_idx = 0

        class MockStdout:
            async def readline(self):
                nonlocal stdout_idx
                if stdout_idx < len(stdout_lines):
                    result = stdout_lines[stdout_idx]
                    stdout_idx += 1
                    return result
                return b""

        class MockStderr:
            async def readline(self):
                nonlocal stderr_idx
                if stderr_idx < len(stderr_lines):
                    result = stderr_lines[stderr_idx]
                    stderr_idx += 1
                    return result
                return b""

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.returncode = return_code
        mock_process.stdout = MockStdout()
        mock_process.stderr = MockStderr()

        async def mock_wait():
            return return_code

        mock_process.wait = mock_wait

        return mock_process

    @pytest.mark.asyncio
    async def test_execute_task_handles_success(self, worker):
        """Test that successful task execution is handled."""
        task = MockTask()

        result_json = json.dumps({
            "type": "result",
            "success": True,
            "data": {"run_id": "test-run"},
        })

        mock_process = self._create_mock_process(
            stdout_lines=[result_json.encode() + b"\n", b""],
            stderr_lines=[b""],
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await worker._execute_task(task)

        worker.queue_manager.complete_task.assert_called_with(
            task.id, {"run_id": "test-run"}
        )

    @pytest.mark.asyncio
    async def test_execute_task_handles_failure(self, worker):
        """Test that task failure is handled."""
        task = MockTask()

        result_json = json.dumps({
            "type": "result",
            "success": False,
            "error": "Something went wrong",
        })

        mock_process = self._create_mock_process(
            stdout_lines=[result_json.encode() + b"\n", b""],
            stderr_lines=[b""],
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await worker._execute_task(task)

        worker.queue_manager.fail_task.assert_called_with(
            task.id, "Something went wrong"
        )

    @pytest.mark.asyncio
    async def test_execute_task_handles_progress(self, worker):
        """Test that progress messages are handled."""
        task = MockTask()

        progress_json = json.dumps({
            "type": "progress",
            "stage": "generating",
            "current": 1,
            "total": 10,
            "message": "Working...",
        })
        result_json = json.dumps({"type": "result", "success": True, "data": {}})

        mock_process = self._create_mock_process(
            stdout_lines=[progress_json.encode() + b"\n", result_json.encode() + b"\n", b""],
            stderr_lines=[b""],
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await worker._execute_task(task)

        worker.queue_manager.update_progress.assert_called()

    @pytest.mark.asyncio
    async def test_execute_task_handles_image_ready(self, worker):
        """Test that image_ready notifications are handled."""
        task = MockTask()

        image_json = json.dumps({
            "type": "image_ready",
            "run_id": "test-run",
            "path": "image_0_0.png",
        })
        result_json = json.dumps({"type": "result", "success": True, "data": {}})

        mock_process = self._create_mock_process(
            stdout_lines=[image_json.encode() + b"\n", result_json.encode() + b"\n", b""],
            stderr_lines=[b""],
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await worker._execute_task(task)

        worker.queue_manager.notify_image_ready.assert_called_with(
            "test-run", "image_0_0.png"
        )

    @pytest.mark.asyncio
    async def test_execute_task_handles_nonzero_exit(self, worker):
        """Test that non-zero exit code is handled."""
        task = MockTask()

        mock_process = self._create_mock_process(
            stdout_lines=[b""],
            stderr_lines=[b"Error occurred\n", b""],
            return_code=1,
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await worker._execute_task(task)

        worker.queue_manager.fail_task.assert_called()

    @pytest.mark.asyncio
    async def test_execute_task_handles_exception(self, worker):
        """Test that exceptions are handled."""
        task = MockTask()

        with patch("asyncio.create_subprocess_exec", side_effect=Exception("Spawn failed")):
            await worker._execute_task(task)

        worker.queue_manager.fail_task.assert_called_with(task.id, "Spawn failed")

    @pytest.mark.asyncio
    async def test_execute_task_handles_invalid_json(self, worker):
        """Test that invalid JSON output is logged but doesn't crash."""
        task = MockTask()

        mock_process = self._create_mock_process(
            stdout_lines=[b"not valid json\n", b""],
            stderr_lines=[b""],
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("logging.warning") as mock_warn:
                await worker._execute_task(task)

                # Should have logged a warning about invalid JSON
                mock_warn.assert_called()


class TestWorkerRun:
    """Tests for Worker.run() main loop."""

    @pytest.mark.asyncio
    async def test_run_stops_when_flag_set(self, worker):
        """Test that run() stops when _running is set to False."""
        call_count = 0

        def mock_get_task():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                worker.stop()
            return None

        worker.queue_manager.get_next_task = mock_get_task

        # Run should exit after stop() is called
        await asyncio.wait_for(worker.run(), timeout=2.0)
        assert call_count >= 3
        assert worker._running is False
