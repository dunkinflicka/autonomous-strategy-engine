"""Structured simulation logging."""
from __future__ import annotations

import logging
import sys
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class SimulationEventLog:
    """
    Lightweight in-memory log of discrete race events.
    Avoids file I/O overhead during Monte Carlo runs.
    """

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def log(self, lap: int, event_type: str, **kwargs: Any) -> None:
        self._events.append({"lap": lap, "type": event_type, **kwargs})

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type is None:
            return list(self._events)
        return [e for e in self._events if e["type"] == event_type]

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)
