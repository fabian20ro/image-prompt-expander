"""FastAPI application for the web UI."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .queue_manager import QueueManager
from .worker import Worker


# Paths
SRC_DIR = Path(__file__).parent.parent
ROOT_DIR = SRC_DIR.parent
GENERATED_DIR = ROOT_DIR / "generated"
QUEUE_PATH = GENERATED_DIR / "queue.json"


# Global instances
queue_manager: QueueManager | None = None
worker: Worker | None = None
shutdown_event: asyncio.Event | None = None


def get_queue_manager() -> QueueManager:
    """Get the queue manager instance."""
    global queue_manager
    if queue_manager is None:
        raise RuntimeError("Queue manager not initialized")
    return queue_manager


def get_worker() -> Worker:
    """Get the worker instance."""
    global worker
    if worker is None:
        raise RuntimeError("Worker not initialized")
    return worker


def get_shutdown_event() -> asyncio.Event:
    """Get the shutdown event."""
    global shutdown_event
    if shutdown_event is None:
        raise RuntimeError("Shutdown event not initialized")
    return shutdown_event


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle app startup and shutdown."""
    global queue_manager, worker, shutdown_event

    # Ensure generated directory exists
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (GENERATED_DIR / "prompts").mkdir(exist_ok=True)
    (GENERATED_DIR / "grammars").mkdir(exist_ok=True)

    # Initialize shutdown event
    shutdown_event = asyncio.Event()

    # Initialize queue manager
    queue_manager = QueueManager(QUEUE_PATH)

    # Initialize and start worker
    worker = Worker(queue_manager, GENERATED_DIR)
    worker_task = asyncio.create_task(worker.run())

    yield

    # Shutdown: signal SSE connections to close
    shutdown_event.set()
    await asyncio.sleep(0.5)  # Grace period for SSE connections to close

    # Stop worker
    worker.stop()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Image Prompt Generator",
        description="Web UI for generating image prompts",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Import and include routes
    from .routes import router
    app.include_router(router)

    # Mount static files for generated content
    # This must be done after routes to not shadow API endpoints
    @app.on_event("startup")
    async def mount_static():
        if GENERATED_DIR.exists():
            app.mount(
                "/generated",
                StaticFiles(directory=str(GENERATED_DIR)),
                name="generated",
            )

    # Redirect root to index
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/index")

    return app


# Create the app instance
app = create_app()
