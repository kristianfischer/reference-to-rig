"""FastAPI application entry point."""

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import projects
from app.config import settings
from app.observability.logging import setup_logging
from app.storage.database import init_db
from app.tasks.queue import task_queue

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    setup_logging(settings.log_level, settings.log_format)
    settings.ensure_directories()
    await init_db()
    task_queue.start()
    logger.info("Application started", version="0.1.0")

    yield

    # Shutdown
    task_queue.stop()
    logger.info("Application shutdown")


app = FastAPI(
    title="Reference-to-Rig Audio Engine",
    description="Local guitar tone matching and rig recipe generator",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "http://127.0.0.1:1420", "tauri://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_correlation_id(request, call_next):
    """Add correlation ID to each request."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


# Include routers
app.include_router(projects.router, prefix="/projects", tags=["projects"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


