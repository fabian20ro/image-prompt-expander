"""Pydantic models for the web server API."""

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of tasks that can be queued."""
    GENERATE_PIPELINE = "generate_pipeline"
    REGENERATE_PROMPTS = "regenerate_prompts"
    GENERATE_IMAGE = "generate_image"
    ENHANCE_IMAGE = "enhance_image"
    GENERATE_ALL_IMAGES = "generate_all_images"
    ENHANCE_ALL_IMAGES = "enhance_all_images"


class TaskStatus(str, Enum):
    """Status of a queued task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskProgress(BaseModel):
    """Progress information for a running task."""
    stage: str = ""
    current: int = 0
    total: int = 0
    message: str = ""


class Task(BaseModel):
    """A task in the queue."""
    id: str
    type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    pid: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    progress: TaskProgress = Field(default_factory=TaskProgress)
    result: dict[str, Any] | None = None
    error: str | None = None


class QueueState(BaseModel):
    """State of the task queue persisted to disk."""
    version: int = 1
    current_task: Task | None = None
    pending: list[Task] = Field(default_factory=list)
    completed: list[Task] = Field(default_factory=list)


# API Request Models

class GenerateRequest(BaseModel):
    """Request to start a new generation pipeline."""
    prompt: str
    count: int = 50
    prefix: str = "image"
    model: str = "z-image-turbo"
    temperature: float = 0.7
    no_cache: bool = False
    generate_images: bool = False
    images_per_prompt: int = 1
    width: int = 864
    height: int = 1152
    steps: int | None = None
    quantize: int = 8
    seed: int | None = None
    max_prompts: int | None = None
    tiled_vae: bool = True
    enhance: bool = False
    enhance_softness: float = 0.5
    enhance_after: bool = False


class RegeneratePromptsRequest(BaseModel):
    """Request to regenerate prompts from edited grammar."""
    run_id: str
    grammar: str
    count: int | None = None


class GrammarUpdateRequest(BaseModel):
    """Request to update a gallery's grammar."""
    grammar: str


class GenerateImageRequest(BaseModel):
    """Request to generate a specific image."""
    image_idx: int = 0


class EnhanceImageRequest(BaseModel):
    """Request to enhance a specific image."""
    image_idx: int = 0
    softness: float = 0.5


class GenerateAllImagesRequest(BaseModel):
    """Request to generate all images for a gallery."""
    images_per_prompt: int = 1
    resume: bool = True


class EnhanceAllImagesRequest(BaseModel):
    """Request to enhance all images for a gallery."""
    softness: float = 0.5


# API Response Models

class StatusResponse(BaseModel):
    """Response for queue status endpoint."""
    queue_length: int
    current_task: Task | None
    pending_count: int
    completed_count: int


class TaskResponse(BaseModel):
    """Response after creating a task."""
    task_id: str
    message: str


class GalleryInfo(BaseModel):
    """Information about a gallery run."""
    run_id: str
    prefix: str
    user_prompt: str
    prompt_count: int
    image_count: int
    created_at: str
    model: str | None
    gallery_path: str
    thumbnail: str | None


class GalleryDetailResponse(BaseModel):
    """Detailed gallery information including grammar."""
    info: GalleryInfo
    grammar: str | None
    prompts: list[str]
    images: list[dict[str, Any]]
