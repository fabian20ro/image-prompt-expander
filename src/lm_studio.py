"""LM Studio process lifecycle helpers."""

import shutil
import subprocess


class LMStudioUnloadError(RuntimeError):
    """Raised when loaded LM Studio models cannot be released."""


def unload_all_models(timeout: float = 60.0) -> None:
    """Unload every LM Studio model before an mflux model is loaded."""
    executable = shutil.which("lms")
    if executable is None:
        raise LMStudioUnloadError("LM Studio CLI `lms` was not found on PATH")

    try:
        result = subprocess.run(
            [executable, "unload", "--all"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LMStudioUnloadError("Timed out unloading LM Studio models") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise LMStudioUnloadError(f"Failed to unload LM Studio models: {detail}")
