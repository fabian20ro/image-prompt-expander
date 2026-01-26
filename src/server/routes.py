"""API routes for the web server."""

import asyncio
import json
import logging
import mimetypes
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from sse_starlette.sse import EventSourceResponse

from .app import get_queue_manager, get_worker, GENERATED_DIR
from .models import (
    GenerateRequest,
    RegeneratePromptsRequest,
    GrammarUpdateRequest,
    GenerateImageRequest,
    EnhanceImageRequest,
    GenerateAllImagesRequest,
    EnhanceAllImagesRequest,
    StatusResponse,
    TaskResponse,
    TaskType,
    GalleryInfo,
    GalleryDetailResponse,
)

# Import gallery index generation
import sys
SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))
from gallery_index import generate_master_index, _extract_run_info


router = APIRouter()


# ----------------------------------------------------------------------------
# Global Endpoints
# ----------------------------------------------------------------------------

@router.get("/index", response_class=HTMLResponse)
async def get_index():
    """Serve the master index page with generation form."""
    # Regenerate index with interactive mode for the web UI
    index_path = generate_master_index(GENERATED_DIR, interactive=True)

    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    else:
        return HTMLResponse(content="<h1>No galleries yet</h1>")


@router.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current queue status."""
    qm = get_queue_manager()
    state = qm.get_state()

    return StatusResponse(
        queue_length=len(state.pending) + (1 if state.current_task else 0),
        current_task=state.current_task,
        pending_count=len(state.pending),
        completed_count=len(state.completed),
    )


@router.get("/api/events")
async def sse_events(request: Request):
    """SSE endpoint for real-time updates."""
    queue = asyncio.Queue(maxsize=100)
    qm = get_queue_manager()

    def on_event(event: str, data: dict):
        try:
            queue.put_nowait({"event": event, "data": data})
        except asyncio.QueueFull:
            logging.warning(f"SSE queue full, dropped event: {event}")

    # Register listener BEFORE getting initial state to avoid race condition
    qm.add_listener(on_event)

    def serialize(obj):
        """Serialize object to JSON, handling datetime and Pydantic models."""
        if hasattr(obj, 'model_dump'):
            return obj.model_dump(mode='json')
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(item) for item in obj]
        return obj

    async def event_generator() -> AsyncGenerator:
        try:
            # Send initial status AFTER registering listener (atomic snapshot)
            state = qm.get_state()
            yield {
                "event": "status",
                "data": json.dumps({
                    "pending_count": len(state.pending),
                    "current": state.current_task.model_dump(mode='json') if state.current_task else None,
                }),
            }

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    yield {
                        "event": msg["event"],
                        "data": json.dumps(serialize(msg["data"])),
                    }
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}

        finally:
            qm.remove_listener(on_event)

    return EventSourceResponse(event_generator())


@router.post("/api/queue/clear", response_model=TaskResponse)
async def clear_queue():
    """Clear all pending tasks."""
    qm = get_queue_manager()
    count = qm.clear_pending()
    return TaskResponse(task_id="", message=f"Cleared {count} pending tasks")


@router.post("/api/worker/kill", response_model=TaskResponse)
async def kill_worker():
    """Kill the currently running task."""
    worker = get_worker()
    killed = await worker.kill_current()

    if killed:
        return TaskResponse(task_id="", message="Killed current task")
    else:
        return TaskResponse(task_id="", message="No task running")


# ----------------------------------------------------------------------------
# Generation Endpoints
# ----------------------------------------------------------------------------

@router.post("/api/generate", response_model=TaskResponse)
async def start_generation(req: GenerateRequest):
    """Start a new generation pipeline."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.GENERATE_PIPELINE,
        req.model_dump(),
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Generation queued (task {task.id[:8]})",
    )


# ----------------------------------------------------------------------------
# Gallery Endpoints
# ----------------------------------------------------------------------------

@router.get("/gallery/{run_id}", response_class=HTMLResponse)
async def get_gallery(run_id: str):
    """Serve a gallery page with interactive features."""
    from gallery import generate_gallery_for_directory

    prompts_dir = GENERATED_DIR / "prompts"
    run_dir = prompts_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    # Regenerate gallery with interactive mode
    try:
        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        return HTMLResponse(content=gallery_path.read_text())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/gallery/{run_id}/{filename:path}")
async def get_gallery_file(run_id: str, filename: str):
    """Serve static files (images, etc.) from a gallery directory."""
    prompts_dir = GENERATED_DIR / "prompts"
    run_dir = prompts_dir / run_id
    file_path = run_dir / filename

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security: ensure file is within the run directory
    try:
        file_path.resolve().relative_to(run_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    # Determine media type
    media_type, _ = mimetypes.guess_type(str(file_path))
    if media_type is None:
        media_type = "application/octet-stream"

    return FileResponse(file_path, media_type=media_type)


@router.get("/api/gallery/{run_id}")
async def get_gallery_info(run_id: str) -> GalleryDetailResponse:
    """Get gallery details including grammar and prompts."""
    prompts_dir = GENERATED_DIR / "prompts"
    run_dir = prompts_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    info = _extract_run_info(run_dir)
    if not info:
        raise HTTPException(status_code=404, detail="Invalid gallery")

    prefix = info["prefix"]

    # Load grammar
    grammar = None
    grammar_file = run_dir / f"{prefix}_grammar.json"
    if grammar_file.exists():
        grammar = grammar_file.read_text()

    # Load prompts
    prompt_files = sorted(run_dir.glob(f"{prefix}_*.txt"))
    prompt_files = [f for f in prompt_files if f.stem.count('_') == 1]
    prompts = [f.read_text() for f in prompt_files]

    # List images
    images = []
    for prompt_idx, prompt_text in enumerate(prompts):
        for image_idx in range(10):  # Check first 10 images per prompt
            img_path = run_dir / f"{prefix}_{prompt_idx}_{image_idx}.png"
            if img_path.exists():
                images.append({
                    "prompt_idx": prompt_idx,
                    "image_idx": image_idx,
                    "filename": img_path.name,
                    "prompt": prompt_text,
                })

    return GalleryDetailResponse(
        info=GalleryInfo(
            run_id=run_id,
            prefix=prefix,
            user_prompt=info["user_prompt"],
            prompt_count=info["prompt_count"],
            image_count=info["image_count"],
            created_at=info["display_time"],
            model=info["model"],
            gallery_path=info["gallery_path"],
            thumbnail=info["thumbnail"],
        ),
        grammar=grammar,
        prompts=prompts,
        images=images,
    )


@router.get("/api/gallery/{run_id}/grammar")
async def get_grammar(run_id: str):
    """Get the grammar JSON for a gallery."""
    prompts_dir = GENERATED_DIR / "prompts"
    run_dir = prompts_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    meta_files = list(run_dir.glob("*_metadata.json"))
    if not meta_files:
        raise HTTPException(status_code=404, detail="No metadata found")

    metadata = json.loads(meta_files[0].read_text())
    prefix = metadata.get("prefix", "image")

    grammar_file = run_dir / f"{prefix}_grammar.json"
    if not grammar_file.exists():
        raise HTTPException(status_code=404, detail="Grammar not found")

    return {"grammar": grammar_file.read_text()}


@router.put("/api/gallery/{run_id}/grammar", response_model=TaskResponse)
async def update_grammar(run_id: str, req: GrammarUpdateRequest):
    """Update the grammar for a gallery."""
    prompts_dir = GENERATED_DIR / "prompts"
    run_dir = prompts_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    # Validate JSON
    try:
        json.loads(req.grammar)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON grammar")

    meta_files = list(run_dir.glob("*_metadata.json"))
    if not meta_files:
        raise HTTPException(status_code=404, detail="No metadata found")

    metadata = json.loads(meta_files[0].read_text())
    prefix = metadata.get("prefix", "image")

    grammar_file = run_dir / f"{prefix}_grammar.json"
    grammar_file.write_text(req.grammar)

    return TaskResponse(task_id="", message="Grammar updated")


@router.post("/api/gallery/{run_id}/regenerate", response_model=TaskResponse)
async def regenerate_prompts(run_id: str, req: RegeneratePromptsRequest | None = None):
    """Regenerate prompts from the current grammar."""
    prompts_dir = GENERATED_DIR / "prompts"
    run_dir = prompts_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    meta_files = list(run_dir.glob("*_metadata.json"))
    if not meta_files:
        raise HTTPException(status_code=404, detail="No metadata found")

    metadata = json.loads(meta_files[0].read_text())
    prefix = metadata.get("prefix", "image")

    grammar_file = run_dir / f"{prefix}_grammar.json"
    if not grammar_file.exists():
        raise HTTPException(status_code=404, detail="Grammar not found")

    grammar = grammar_file.read_text()
    count = req.count if req and req.count else metadata.get("count", 50)

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.REGENERATE_PROMPTS,
        {
            "run_id": run_id,
            "grammar": grammar,
            "count": count,
        },
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Prompt regeneration queued (task {task.id[:8]})",
    )


@router.post("/api/gallery/{run_id}/generate-all", response_model=TaskResponse)
async def generate_all_images(run_id: str, req: GenerateAllImagesRequest | None = None):
    """Queue generation of all images for a gallery."""
    prompts_dir = GENERATED_DIR / "prompts"
    run_dir = prompts_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.GENERATE_ALL_IMAGES,
        {
            "run_id": run_id,
            "images_per_prompt": req.images_per_prompt if req else 1,
            "resume": req.resume if req else True,
        },
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Image generation queued (task {task.id[:8]})",
    )


@router.post("/api/gallery/{run_id}/enhance-all", response_model=TaskResponse)
async def enhance_all_images(run_id: str, req: EnhanceAllImagesRequest | None = None):
    """Queue enhancement of all images for a gallery."""
    prompts_dir = GENERATED_DIR / "prompts"
    run_dir = prompts_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.ENHANCE_ALL_IMAGES,
        {
            "run_id": run_id,
            "softness": req.softness if req else 0.5,
        },
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Enhancement queued (task {task.id[:8]})",
    )


@router.post("/api/gallery/{run_id}/image/{prompt_idx}/generate", response_model=TaskResponse)
async def generate_single_image(
    run_id: str,
    prompt_idx: int,
    req: GenerateImageRequest | None = None,
):
    """Queue generation of a single image."""
    prompts_dir = GENERATED_DIR / "prompts"
    run_dir = prompts_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.GENERATE_IMAGE,
        {
            "run_id": run_id,
            "prompt_idx": prompt_idx,
            "image_idx": req.image_idx if req else 0,
        },
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Image generation queued (task {task.id[:8]})",
    )


@router.post("/api/gallery/{run_id}/image/{prompt_idx}/enhance", response_model=TaskResponse)
async def enhance_single_image(
    run_id: str,
    prompt_idx: int,
    req: EnhanceImageRequest | None = None,
):
    """Queue enhancement of a single image."""
    prompts_dir = GENERATED_DIR / "prompts"
    run_dir = prompts_dir / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Gallery not found")

    qm = get_queue_manager()
    task = qm.add_task(
        TaskType.ENHANCE_IMAGE,
        {
            "run_id": run_id,
            "prompt_idx": prompt_idx,
            "image_idx": req.image_idx if req else 0,
            "softness": req.softness if req else 0.5,
        },
    )

    return TaskResponse(
        task_id=task.id,
        message=f"Enhancement queued (task {task.id[:8]})",
    )
