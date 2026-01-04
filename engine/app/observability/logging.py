"""Structured logging configuration."""

import logging
import sys
from typing import Literal

import structlog


def setup_logging(
    level: str = "INFO",
    format_type: Literal["json", "console"] = "console",
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: Output format (json for production, console for dev)
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Reduce noise from libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Configure structlog
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format_type == "json":
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, logger=None):
        self.name = name
        self.logger = logger or structlog.get_logger()
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        self.elapsed = time.perf_counter() - self.start_time
        self.logger.debug(
            "Timer completed",
            timer=self.name,
            elapsed_ms=round(self.elapsed * 1000, 2),
        )


