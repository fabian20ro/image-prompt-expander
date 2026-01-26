"""Centralized configuration for the image-prompt-expander application."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LMStudioConfig:
    """Configuration for LM Studio connection."""
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"


@dataclass(frozen=True)
class ImageGenerationConfig:
    """Default configuration for image generation."""
    default_width: int = 864
    default_height: int = 1152
    default_steps: int = 4
    default_quantize: int = 8
    default_model: str = "z-image-turbo"


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the web server."""
    sse_queue_size: int = 100
    sse_timeout: float = 5.0  # seconds between keepalives
    worker_timeout: float = 300.0  # seconds


@dataclass(frozen=True)
class EnhancementConfig:
    """Configuration for image enhancement."""
    default_softness: float = 0.5
    default_scale: int = 2


@dataclass
class Settings:
    """Application settings, can be overridden via environment variables."""
    lm_studio: LMStudioConfig = field(default_factory=LMStudioConfig)
    image_generation: ImageGenerationConfig = field(default_factory=ImageGenerationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables with PROMPT_GEN_ prefix."""
        lm_studio = LMStudioConfig(
            base_url=os.environ.get("PROMPT_GEN_LM_STUDIO_URL", LMStudioConfig.base_url),
            api_key=os.environ.get("PROMPT_GEN_LM_STUDIO_API_KEY", LMStudioConfig.api_key),
        )
        image_generation = ImageGenerationConfig(
            default_width=int(os.environ.get("PROMPT_GEN_DEFAULT_WIDTH", ImageGenerationConfig.default_width)),
            default_height=int(os.environ.get("PROMPT_GEN_DEFAULT_HEIGHT", ImageGenerationConfig.default_height)),
            default_steps=int(os.environ.get("PROMPT_GEN_DEFAULT_STEPS", ImageGenerationConfig.default_steps)),
            default_quantize=int(os.environ.get("PROMPT_GEN_DEFAULT_QUANTIZE", ImageGenerationConfig.default_quantize)),
            default_model=os.environ.get("PROMPT_GEN_DEFAULT_MODEL", ImageGenerationConfig.default_model),
        )
        server = ServerConfig(
            sse_queue_size=int(os.environ.get("PROMPT_GEN_SSE_QUEUE_SIZE", ServerConfig.sse_queue_size)),
            sse_timeout=float(os.environ.get("PROMPT_GEN_SSE_TIMEOUT", ServerConfig.sse_timeout)),
            worker_timeout=float(os.environ.get("PROMPT_GEN_WORKER_TIMEOUT", ServerConfig.worker_timeout)),
        )
        enhancement = EnhancementConfig(
            default_softness=float(os.environ.get("PROMPT_GEN_ENHANCE_SOFTNESS", EnhancementConfig.default_softness)),
            default_scale=int(os.environ.get("PROMPT_GEN_ENHANCE_SCALE", EnhancementConfig.default_scale)),
        )
        return cls(
            lm_studio=lm_studio,
            image_generation=image_generation,
            server=server,
            enhancement=enhancement,
        )


# Global settings instance - use from_env() for environment-aware settings
settings = Settings.from_env()
