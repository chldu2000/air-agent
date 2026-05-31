from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from air_agent.types import RunEvent


logger = logging.getLogger(__name__)

EventHandler = Callable[[RunEvent], Any]


class EventDispatcher:
    def __init__(
        self,
        enabled: bool = False,
        handlers: list[EventHandler] | None = None,
        log_events: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.enabled = enabled
        self.handlers = handlers or []
        self.log_events = log_events
        self.logger = logger or logging.getLogger(__name__)

    async def emit(self, event: RunEvent) -> None:
        if not self.enabled:
            return

        if event.timestamp is None:
            event.timestamp = datetime.now(timezone.utc)

        if self.log_events:
            try:
                self.logger.info(json.dumps(event.to_dict(), ensure_ascii=False))
            except Exception:
                self.logger.warning("Failed to log run event", exc_info=True)

        for handler in self.handlers:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                self.logger.warning("Run event handler failed", exc_info=True)
