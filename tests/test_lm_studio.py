"""Tests for src/lm_studio."""

from unittest.mock import patch, MagicMock
import subprocess
import pytest

from lm_studio import unload_all_models, LMStudioUnloadError


class TestUnloadAllModels:
    def test_raises_when_lms_not_found(self):
        with patch("lm_studio.shutil.which", return_value=None):
            with pytest.raises(LMStudioUnloadError) as exc_info:
                unload_all_models()
            assert "not found on PATH" in str(exc_info.value)

    def test_succeeds_on_first_attempt(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run = MagicMock(return_value=mock_result)

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", mock_run) as run_mock:
            unload_all_models()
            assert run_mock.call_count == 1

    def test_retries_on_failure(self):
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "still busy"
        success_result = MagicMock()
        success_result.returncode = 0

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return fail_result
            return success_result

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", side_effect=side_effect), \
             patch("lm_studio.time.sleep") as sleep_mock:
            unload_all_models()
            assert call_count[0] == 3
            assert sleep_mock.call_count == 2

    def test_raises_after_exhausting_retries(self):
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "busy"

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", return_value=fail_result), \
             patch("lm_studio.time.sleep"):
            with pytest.raises(LMStudioUnloadError) as exc_info:
                unload_all_models()
            assert "busy" in str(exc_info.value)

    def test_raises_on_timeout_after_exhausting_retries(self):
        timeout_exc = subprocess.TimeoutExpired(cmd=["lms", "unload", "--all"], timeout=60)

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise timeout_exc
            return MagicMock(returncode=0)

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", side_effect=side_effect), \
             patch("lm_studio.time.sleep"):
            unload_all_models()
            assert call_count[0] == 3

    def test_timeout_on_final_attempt_raises(self):
        timeout_exc = subprocess.TimeoutExpired(cmd=["lms", "unload", "--all"], timeout=60)

        with patch("lm_studio.shutil.which", return_value="/usr/bin/lms"), \
             patch("lm_studio.subprocess.run", side_effect=lambda *a, **kw: (_ for _ in ()).throw(timeout_exc)), \
             patch("lm_studio.time.sleep"):
            with pytest.raises(LMStudioUnloadError) as exc_info:
                unload_all_models()
            assert "Timed out" in str(exc_info.value)
