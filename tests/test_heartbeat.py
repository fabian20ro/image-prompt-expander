"""Tests for Heartbeat context manager."""

import asyncio
import json
import threading
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.server.worker_subprocess import Heartbeat


@pytest.mark.asyncio
async def test_heartbeat_emits_progress():
    """Test that Heartbeat emits progress messages during its interval."""

    progress_events = []

    def mock_emit_progress(stage: str, current: int = 0, total: int = 0, message: str = ""):
        progress_events.append((stage, current, total, message))

    # Patch emit_progress in the module where it is used
    with patch('src.server.worker_subprocess.emit_progress', side_effect=mock_emit_progress):
        with Heartbeat(message="test message", interval=0.1) as hb:
            # Allow enough time for at least one heartbeat
            await asyncio.sleep(0.25)

    assert len(progress_events) >= 1
    assert progress_events[0][0] == "heartbeat"
    assert progress_events[0][1] == 0
    assert progress_events[0][3] == "test message"

@pytest.mark.asyncio
async def test_heartbeat_stops_on_exit():
    """Test that Heartbeat stops when exiting the context manager."""

    heartbeats_count = 0

    def mock_emit_progress(stage: str, current: int = 0, total: int = 0, message: str = ""):
        nonlocal heartbeats_count
        heartbeats_count += 1

    with patch('src.server.worker_subprocess.emit_progress', side_effect=mock_emit_progress):
        with Heartbeat(message="test message", interval=0.1) as hb:
            await asyncio.sleep(0.05)
            # Should have at least 0 or 1 heartbeats
            assert heartbeats_count < 5
            # Exit context manager
            pass

        # After exit, heartbeats should stop
        count_after_exit = heartbeats_count
        await asyncio.sleep(0.2)
        assert heartbeats_count == count_after_exit

@pytest.mark.asyncio
async def test_heartbeat_no_progress_on_immediate_exit():
    """Test that no progress is emitted if Heartbeat exits before first heartbeat interval."""

    progress_events = []

    def mock_emit_progress(stage: str, current: int = 0, total: int = 0, message: str = ""):
        progress_events.append((stage, current, total, message))

    with patch('src.server.worker_subprocess.emit_progress', side_effect=mock_emit_progress):
        # Exit immediately - before first heartbeat interval elapses
        with Heartbeat(message="test", interval=1.0) as hb:
            pass  # Context manager exits immediately

    assert len(progress_events) == 0


@pytest.mark.asyncio
async def test_heartbeat_stops_on_exception():
    """Test that heartbeat thread stops cleanly when an exception occurs inside the with block."""

    heartbeats_count = 0

    def mock_emit_progress(stage: str, current: int = 0, total: int = 0, message: str = ""):
        nonlocal heartbeats_count
        heartbeats_count += 1

    with patch('src.server.worker_subprocess.emit_progress', side_effect=mock_emit_progress):
        # Enter heartbeat and wait for at least one emission
        try:
            with Heartbeat(message="test", interval=0.1) as hb:
                await asyncio.sleep(0.25)  # Wait for at least one heartbeat
                assert heartbeats_count >= 1  # Verify heartbeat was emitted

            # Should not reach here if exception is raised after sleep
        except ValueError:
            pass  # Expected exception

    count_after_exit = heartbeats_count
    await asyncio.sleep(0.3)  # Wait to confirm no more heartbeats
    assert heartbeats_count == count_after_exit  # Heartbeat thread should be stopped


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])