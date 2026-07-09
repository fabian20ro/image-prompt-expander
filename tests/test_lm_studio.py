"""Tests for the LM Studio memory handoff."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from lm_studio import LMStudioUnloadError, unload_all_models


@patch("lm_studio.subprocess.run")
@patch("lm_studio.shutil.which", return_value="/opt/lms")
def test_unload_all_models(mock_which, mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    unload_all_models()
    mock_run.assert_called_once_with(
        ["/opt/lms", "unload", "--all"],
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
    )


@patch("lm_studio.shutil.which", return_value=None)
def test_unload_requires_cli(mock_which):
    with pytest.raises(LMStudioUnloadError, match="not found"):
        unload_all_models()


@patch("lm_studio.subprocess.run")
@patch("lm_studio.shutil.which", return_value="/opt/lms")
def test_unload_fails_closed(mock_which, mock_run):
    mock_run.return_value = MagicMock(returncode=1, stderr="busy", stdout="")
    with pytest.raises(LMStudioUnloadError, match="busy"):
        unload_all_models()


@patch("lm_studio.subprocess.run", side_effect=subprocess.TimeoutExpired("lms", 60))
@patch("lm_studio.shutil.which", return_value="/opt/lms")
def test_unload_timeout(mock_which, mock_run):
    with pytest.raises(LMStudioUnloadError, match="Timed out"):
        unload_all_models()


@patch("lm_studio.subprocess.run")
@patch("lm_studio.shutil.which", return_value="/opt/lms")
def test_unload_fails_open(mock_which, mock_run):
    """Stdout-only error message is surfaced when stderr is empty."""
    mock_run.return_value = MagicMock(returncode=1, stderr="", stdout="model busy")
    with pytest.raises(LMStudioUnloadError, match="model busy"):
        unload_all_models()


@patch("lm_studio.subprocess.run")
@patch("lm_studio.shutil.which", return_value="/opt/lms")
def test_unload_fails_no_detail(mock_which, mock_run):
    """Empty stderr+stdout falls back to the generic message."""
    mock_run.return_value = MagicMock(returncode=1, stderr="", stdout="")
    with pytest.raises(LMStudioUnloadError, match="unknown error"):
        unload_all_models()
