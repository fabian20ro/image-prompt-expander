"""Tests for cli.py - CLI entry point."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from cli import main, clean_generated, cli_progress


class TestCleanGenerated:
    """Tests for the clean_generated function."""

    def test_clean_removes_files(self, temp_dir):
        """Test that clean removes generated files."""
        grammars = temp_dir / "grammars"
        prompts = temp_dir / "prompts"
        grammars.mkdir()
        prompts.mkdir()

        (grammars / "test.json").write_text("{}")
        (prompts / "run1").mkdir()
        (prompts / "run1" / "test.txt").write_text("test")

        with patch("cli.paths") as mock_paths:
            mock_paths.grammars_dir = grammars
            mock_paths.prompts_dir = prompts
            count = clean_generated()

        assert count == 2  # 1 file + 1 directory

    def test_clean_empty_dirs(self, temp_dir):
        """Test clean with empty directories."""
        grammars = temp_dir / "grammars"
        prompts = temp_dir / "prompts"
        grammars.mkdir()
        prompts.mkdir()

        with patch("cli.paths") as mock_paths:
            mock_paths.grammars_dir = grammars
            mock_paths.prompts_dir = prompts
            count = clean_generated()

        assert count == 0

    def test_clean_nonexistent_dirs(self, temp_dir):
        """Test clean when dirs don't exist."""
        with patch("cli.paths") as mock_paths:
            mock_paths.grammars_dir = temp_dir / "nonexistent1"
            mock_paths.prompts_dir = temp_dir / "nonexistent2"
            count = clean_generated()

        assert count == 0


class TestCliProgress:
    """Tests for the cli_progress callback."""

    def test_progress_with_message(self, capsys):
        """Test progress with a message."""
        cli_progress("stage", 1, 10, "Working...")
        captured = capsys.readouterr()
        assert "[1/10] Working..." in captured.out

    def test_progress_without_counts(self, capsys):
        """Test progress with just a message."""
        cli_progress("stage", 0, 0, "Starting...")
        captured = capsys.readouterr()
        assert "Starting..." in captured.out

    def test_progress_empty_message(self, capsys):
        """Test progress with empty message is silent."""
        cli_progress("stage", 1, 10, "")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestCliCleanCommand:
    """Tests for --clean flag."""

    def test_clean_flag(self):
        """Test --clean removes generated files."""
        runner = CliRunner()
        with patch("cli.clean_generated", return_value=5) as mock_clean:
            result = runner.invoke(main, ["--clean"])
            assert result.exit_code == 0
            assert "Cleaned 5 items" in result.output
            mock_clean.assert_called_once()


class TestCliValidation:
    """Tests for CLI argument validation."""

    def test_no_args_shows_error(self):
        """Test that no arguments shows error."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_from_grammar_and_from_prompts_conflict(self, temp_dir):
        """Test conflicting flags."""
        grammar_file = temp_dir / "test.json"
        grammar_file.write_text('{"origin": ["test"]}')
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            "--from-grammar", str(grammar_file),
            "--from-prompts", str(prompts_dir),
        ])
        assert result.exit_code != 0
        assert "Cannot use both" in result.output

    def test_from_prompts_requires_generate_images(self, temp_dir):
        """Test --from-prompts requires --generate-images."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, ["--from-prompts", str(prompts_dir)])
        assert result.exit_code != 0
        assert "--generate-images" in result.output


class TestCliServe:
    """Tests for --serve flag."""

    def test_serve_starts_server(self):
        """Test that --serve starts the web server."""
        runner = CliRunner()

        mock_uvicorn = MagicMock()
        mock_app = MagicMock()

        with patch.dict("sys.modules", {
            "uvicorn": mock_uvicorn,
            "server.app": MagicMock(app=mock_app),
        }):
            result = runner.invoke(main, ["--serve", "--port", "9999"])

        assert "Starting web UI server" in result.output


class TestCliFullPipeline:
    """Tests for full pipeline execution via CLI."""

    @patch("cli.PipelineExecutor")
    def test_prompt_generates_outputs(self, mock_executor_cls):
        """Test that -p runs the full pipeline."""
        mock_executor = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = Path("/tmp/test")
        mock_result.prompt_count = 5
        mock_result.image_count = 0
        mock_result.error = None
        mock_executor.run_full_pipeline.return_value = mock_result
        mock_executor_cls.return_value = mock_executor

        runner = CliRunner()
        result = runner.invoke(main, ["-p", "a cat", "-n", "5"])

        assert result.exit_code == 0
        mock_executor.run_full_pipeline.assert_called_once()
