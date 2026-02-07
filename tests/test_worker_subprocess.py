"""Tests for worker_subprocess.py - isolated task execution."""

import json
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Import after path setup in conftest
from server.worker_subprocess import (
    emit_progress,
    emit_result,
    emit_image_ready,
    Heartbeat,
    create_executor,
    run_generate_pipeline,
    run_regenerate_prompts,
    run_generate_image,
    run_enhance_image,
    run_generate_all_images,
    run_enhance_all_images,
    run_delete_gallery,
    TASK_HANDLERS,
    set_log_file,
    close_log_file,
    log_to_file,
)


class TestEmitFunctions:
    """Tests for JSON output emission functions."""

    def test_emit_progress(self, capsys):
        """Test that emit_progress outputs valid JSON."""
        emit_progress("test_stage", 5, 10, "Test message")

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["type"] == "progress"
        assert data["stage"] == "test_stage"
        assert data["current"] == 5
        assert data["total"] == 10
        assert data["message"] == "Test message"

    def test_emit_result_success(self, capsys):
        """Test that emit_result outputs success JSON."""
        emit_result(True, data={"run_id": "test-run"})

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["type"] == "result"
        assert data["success"] is True
        assert data["data"] == {"run_id": "test-run"}
        assert data["error"] is None

    def test_emit_result_failure(self, capsys):
        """Test that emit_result outputs failure JSON."""
        emit_result(False, error="Something went wrong")

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["type"] == "result"
        assert data["success"] is False
        assert data["error"] == "Something went wrong"

    def test_emit_image_ready(self, capsys):
        """Test that emit_image_ready outputs valid JSON."""
        emit_image_ready("test-run", "image_0_0.png")

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())

        assert data["type"] == "image_ready"
        assert data["run_id"] == "test-run"
        assert data["path"] == "image_0_0.png"


class TestLogFile:
    """Tests for log file functionality."""

    def test_set_and_close_log_file(self, temp_dir):
        """Test setting and closing log file."""
        log_path = temp_dir / "test.log"

        set_log_file(log_path)
        log_to_file("Test message")
        close_log_file()

        assert log_path.exists()
        content = log_path.read_text()
        assert "Test message" in content

    def test_log_to_file_without_file(self):
        """Test that log_to_file works when no file is set."""
        # Should not raise
        close_log_file()  # Ensure no file is open
        log_to_file("Test message")


class TestHeartbeat:
    """Tests for Heartbeat context manager."""

    def test_heartbeat_as_context_manager(self):
        """Test Heartbeat can be used as context manager."""
        with Heartbeat("Working...", interval=60) as hb:
            assert hb is not None
            assert hb._thread is not None
            assert hb._thread.is_alive()

        # Thread should stop after exiting
        assert not hb._thread.is_alive()

    def test_heartbeat_emits_progress(self, capsys):
        """Test Heartbeat emits progress at intervals."""
        import time

        with Heartbeat("Working...", interval=0.1):
            time.sleep(0.25)  # Wait for at least one heartbeat

        captured = capsys.readouterr()
        # Should have emitted at least one heartbeat
        lines = [l for l in captured.out.strip().split('\n') if l]
        heartbeats = [json.loads(l) for l in lines if json.loads(l).get("stage") == "heartbeat"]
        assert len(heartbeats) >= 1


class TestCreateExecutor:
    """Tests for executor creation."""

    def test_create_executor_returns_pipeline_executor(self):
        """Test that create_executor returns a PipelineExecutor."""
        executor = create_executor()

        from pipeline import PipelineExecutor
        assert isinstance(executor, PipelineExecutor)


class TestTaskHandlers:
    """Tests for task handler mapping."""

    def test_all_handlers_registered(self):
        """Test that all expected handlers are registered."""
        expected_handlers = [
            "generate_pipeline",
            "regenerate_prompts",
            "generate_image",
            "enhance_image",
            "generate_all_images",
            "enhance_all_images",
            "delete_gallery",
        ]

        for handler_name in expected_handlers:
            assert handler_name in TASK_HANDLERS
            assert callable(TASK_HANDLERS[handler_name])


class TestRunGeneratePipeline:
    """Tests for run_generate_pipeline handler."""

    @patch("server.worker_subprocess.create_executor")
    @patch("server.worker_subprocess.set_log_file")
    def test_run_generate_pipeline_success(self, mock_log, mock_exec, capsys):
        """Test successful pipeline generation."""
        from pipeline import PipelineResult

        mock_executor = MagicMock()
        mock_executor.run_full_pipeline.return_value = PipelineResult(
            success=True,
            run_id="test-run",
            output_dir=Path("/tmp/test"),
            prompt_count=10,
        )
        mock_exec.return_value = mock_executor

        run_generate_pipeline({
            "prompt": "test prompt",
            "count": 10,
            "prefix": "test",
        })

        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')
        result = json.loads(lines[-1])

        assert result["success"] is True
        assert result["data"]["run_id"] == "test-run"

    def test_run_generate_pipeline_prepares_log_path_and_output_dir(self, capsys):
        """Pipeline generation should prepare log file and pass output_dir to executor."""
        from pipeline import PipelineResult

        expected_output_dir = Path("/tmp/generated/prompts/20260207_140000_abc123")

        with (
            patch("server.worker_subprocess.create_executor") as mock_exec,
            patch("server.worker_subprocess.set_log_file") as mock_set_log,
            patch("server.worker_subprocess.log_to_file") as mock_log_to_file,
            patch("server.worker_subprocess.hash_prompt", return_value="abc123"),
            patch("server.worker_subprocess.paths") as mock_paths,
            patch("server.worker_subprocess.datetime") as mock_datetime,
            patch("server.worker_subprocess.Heartbeat") as mock_heartbeat,
        ):
            mock_paths.prompts_dir = Path("/tmp/generated/prompts")

            mock_now = MagicMock()
            mock_now.strftime.return_value = "20260207_140000"
            mock_datetime.now.return_value = mock_now

            heartbeat_ctx = MagicMock()
            heartbeat_ctx.__enter__.return_value = None
            heartbeat_ctx.__exit__.return_value = False
            mock_heartbeat.return_value = heartbeat_ctx

            mock_executor = MagicMock()

            def run_full_pipeline_side_effect(**kwargs):
                # Ensure logging was configured before execution starts.
                assert mock_set_log.called
                assert kwargs["output_dir"] == expected_output_dir
                return PipelineResult(
                    success=True,
                    run_id="test-run",
                    output_dir=kwargs["output_dir"],
                    prompt_count=3,
                )

            mock_executor.run_full_pipeline.side_effect = run_full_pipeline_side_effect
            mock_exec.return_value = mock_executor

            run_generate_pipeline({
                "prompt": "test prompt",
                "count": 3,
                "prefix": "test",
            })

            mock_set_log.assert_called_once_with(expected_output_dir / "test_worker.log")
            assert mock_log_to_file.call_count >= 1
            first_log_message = mock_log_to_file.call_args_list[0].args[0]
            assert "Starting generation pipeline" in first_log_message

            called_kwargs = mock_executor.run_full_pipeline.call_args.kwargs
            assert called_kwargs["output_dir"] == expected_output_dir

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]
        result = json.loads(lines[-1])
        assert result["success"] is True
        assert result["data"]["output_dir"] == str(expected_output_dir)

    @patch("server.worker_subprocess.create_executor")
    def test_run_generate_pipeline_failure(self, mock_exec, capsys):
        """Test pipeline generation failure."""
        from pipeline import PipelineResult

        mock_executor = MagicMock()
        mock_executor.run_full_pipeline.return_value = PipelineResult(
            success=False,
            error="Grammar generation failed",
        )
        mock_exec.return_value = mock_executor

        run_generate_pipeline({"prompt": "test"})

        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')
        result = json.loads(lines[-1])

        assert result["success"] is False
        assert "Grammar generation failed" in result["error"]


class TestRunRegeneratePrompts:
    """Tests for run_regenerate_prompts handler."""

    @patch("server.worker_subprocess.create_executor")
    @patch("server.worker_subprocess.set_log_file")
    @patch("server.worker_subprocess.paths")
    def test_run_regenerate_prompts_success(self, mock_paths, mock_log, mock_exec, capsys, temp_dir):
        """Test successful prompt regeneration."""
        from pipeline import PipelineResult
        from conftest import create_run_files

        # Set up mock paths
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir(parents=True)
        run_dir = prompts_dir / "test-run"
        run_dir.mkdir(parents=True)
        create_run_files(run_dir, prefix="test", num_prompts=2)
        mock_paths.prompts_dir = prompts_dir

        mock_executor = MagicMock()
        mock_executor.regenerate_prompts.return_value = PipelineResult(
            success=True,
            run_id="test-run",
            prompt_count=5,
        )
        mock_exec.return_value = mock_executor

        run_regenerate_prompts({
            "run_id": "test-run",
            "grammar": '{"origin": ["test"]}',
            "count": 5,
        })

        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')
        result = json.loads(lines[-1])

        assert result["success"] is True


class TestRunGenerateImage:
    """Tests for run_generate_image handler."""

    @patch("server.worker_subprocess.create_executor")
    @patch("server.worker_subprocess.set_log_file")
    @patch("server.worker_subprocess.paths")
    def test_run_generate_image_success(self, mock_paths, mock_log, mock_exec, capsys, temp_dir):
        """Test successful single image generation."""
        from pipeline import PipelineResult
        from conftest import create_run_files

        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir(parents=True)
        run_dir = prompts_dir / "test-run"
        run_dir.mkdir(parents=True)
        create_run_files(run_dir, prefix="test", num_prompts=2)
        mock_paths.prompts_dir = prompts_dir

        mock_executor = MagicMock()
        mock_executor.generate_single_image.return_value = PipelineResult(
            success=True,
            run_id="test-run",
            data={"image_path": "test_0_0.png"},
        )
        mock_exec.return_value = mock_executor

        run_generate_image({
            "run_id": "test-run",
            "prompt_idx": 0,
            "image_idx": 0,
        })

        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')
        result = json.loads(lines[-1])

        assert result["success"] is True
        assert result["data"]["image_path"] == "test_0_0.png"


class TestRunEnhanceImage:
    """Tests for run_enhance_image handler."""

    @patch("server.worker_subprocess.create_executor")
    @patch("server.worker_subprocess.set_log_file")
    @patch("server.worker_subprocess.paths")
    def test_run_enhance_image_success(self, mock_paths, mock_log, mock_exec, capsys, temp_dir):
        """Test successful single image enhancement."""
        from pipeline import PipelineResult
        from conftest import create_run_files

        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir(parents=True)
        run_dir = prompts_dir / "test-run"
        run_dir.mkdir(parents=True)
        create_run_files(run_dir, prefix="test", num_prompts=2, create_images=True)
        mock_paths.prompts_dir = prompts_dir

        mock_executor = MagicMock()
        mock_executor.enhance_single_image.return_value = PipelineResult(
            success=True,
            run_id="test-run",
            data={"image_path": "test_0_0.png"},
        )
        mock_exec.return_value = mock_executor

        run_enhance_image({
            "run_id": "test-run",
            "prompt_idx": 0,
            "image_idx": 0,
            "softness": 0.5,
        })

        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')
        result = json.loads(lines[-1])

        assert result["success"] is True


class TestRunGenerateAllImages:
    """Tests for run_generate_all_images handler."""

    @patch("server.worker_subprocess.create_executor")
    @patch("server.worker_subprocess.set_log_file")
    @patch("server.worker_subprocess.paths")
    def test_run_generate_all_images_success(self, mock_paths, mock_log, mock_exec, capsys, temp_dir):
        """Test successful batch image generation."""
        from pipeline import PipelineResult
        from conftest import create_run_files

        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir(parents=True)
        run_dir = prompts_dir / "test-run"
        run_dir.mkdir(parents=True)
        create_run_files(run_dir, prefix="test", num_prompts=3)
        mock_paths.prompts_dir = prompts_dir

        mock_executor = MagicMock()
        mock_executor.generate_all_images.return_value = PipelineResult(
            success=True,
            run_id="test-run",
            image_count=3,
            skipped_count=0,
        )
        mock_exec.return_value = mock_executor

        run_generate_all_images({
            "run_id": "test-run",
            "images_per_prompt": 1,
            "resume": True,
        })

        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')
        result = json.loads(lines[-1])

        assert result["success"] is True
        assert result["data"]["generated"] == 3


class TestRunEnhanceAllImages:
    """Tests for run_enhance_all_images handler."""

    @patch("server.worker_subprocess.create_executor")
    @patch("server.worker_subprocess.set_log_file")
    @patch("server.worker_subprocess.paths")
    def test_run_enhance_all_images_success(self, mock_paths, mock_log, mock_exec, capsys, temp_dir):
        """Test successful batch image enhancement."""
        from pipeline import PipelineResult
        from conftest import create_run_files

        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir(parents=True)
        run_dir = prompts_dir / "test-run"
        run_dir.mkdir(parents=True)
        create_run_files(run_dir, prefix="test", num_prompts=3, create_images=True)
        mock_paths.prompts_dir = prompts_dir

        mock_executor = MagicMock()
        mock_executor.enhance_all_images.return_value = PipelineResult(
            success=True,
            run_id="test-run",
            image_count=3,
        )
        mock_exec.return_value = mock_executor

        run_enhance_all_images({
            "run_id": "test-run",
            "softness": 0.5,
        })

        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')
        result = json.loads(lines[-1])

        assert result["success"] is True
        assert result["data"]["enhanced"] == 3


class TestRunDeleteGallery:
    """Tests for run_delete_gallery handler."""

    @patch("server.worker_subprocess.delete_run")
    @patch("server.worker_subprocess.generate_master_index")
    @patch("server.worker_subprocess.paths")
    def test_run_delete_gallery_success(self, mock_paths, mock_index, mock_delete, capsys, temp_dir):
        """Test successful gallery deletion."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()
        run_dir = prompts_dir / "test-run"
        run_dir.mkdir()
        mock_paths.prompts_dir = prompts_dir
        mock_paths.generated_dir = temp_dir

        run_delete_gallery({"run_id": "test-run"})

        captured = capsys.readouterr()
        lines = captured.out.strip().split('\n')
        result = json.loads(lines[-1])

        assert result["success"] is True
        assert result["data"]["deleted"] is True
        mock_delete.assert_called_once()
        mock_index.assert_called_once()


class TestMainFunction:
    """Tests for main() entry point."""

    @patch("server.worker_subprocess.TASK_HANDLERS")
    @patch("server.worker_subprocess.close_log_file")
    def test_main_with_valid_task(self, mock_close, mock_handlers, temp_dir):
        """Test main() with a valid task file."""
        # Create task file
        task_file = temp_dir / "task.json"
        task_file.write_text(json.dumps({
            "type": "generate_pipeline",
            "params": {"prompt": "test"},
        }))

        mock_handler = MagicMock()
        mock_handlers.get.return_value = mock_handler

        from server.worker_subprocess import main

        with patch.object(sys, "argv", ["worker_subprocess.py", str(task_file)]):
            main()

        mock_handler.assert_called_once_with({"prompt": "test"})
        mock_close.assert_called_once()

    def test_main_missing_args(self, capsys):
        """Test main() with missing arguments."""
        from server.worker_subprocess import main

        with patch.object(sys, "argv", ["worker_subprocess.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    def test_main_missing_task_file(self, capsys, temp_dir):
        """Test main() with non-existent task file."""
        from server.worker_subprocess import main

        with patch.object(sys, "argv", ["worker_subprocess.py", str(temp_dir / "nonexistent.json")]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1

    def test_main_invalid_json(self, capsys, temp_dir):
        """Test main() with invalid JSON in task file."""
        task_file = temp_dir / "task.json"
        task_file.write_text("{invalid json")

        from server.worker_subprocess import main

        with patch.object(sys, "argv", ["worker_subprocess.py", str(task_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
