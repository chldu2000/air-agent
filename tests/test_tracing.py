from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest

from air_agent.tracing import EventDispatcher
from air_agent.types import RunEvent, ToolExecutionResult


def test_run_event_accepts_required_observability_fields():
    timestamp = datetime.now(timezone.utc)
    event = RunEvent(
        type="tool_start",
        run_id="run_123",
        conversation_id="conv_1",
        iteration=2,
        timestamp=timestamp,
        name="add",
        arguments='{"a": 1, "b": 2}',
    )

    assert event.type == "tool_start"
    assert event.run_id == "run_123"
    assert event.conversation_id == "conv_1"
    assert event.iteration == 2
    assert event.timestamp == timestamp
    assert event.name == "add"
    assert event.arguments == '{"a": 1, "b": 2}'


def test_run_event_to_dict_serializes_timestamp_and_usage():
    timestamp = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)
    event = RunEvent(
        type="llm_end",
        run_id="run_123",
        timestamp=timestamp,
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )

    data = event.to_dict()

    assert data["type"] == "llm_end"
    assert data["run_id"] == "run_123"
    assert data["timestamp"] == "2026-05-31T12:00:00+00:00"
    assert data["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert "name" not in data


def test_tool_execution_result_success_and_failure_shapes():
    success = ToolExecutionResult.success(content="8", duration_ms=12.5)
    failure = ToolExecutionResult.failure(
        content="Error executing tool 'add': boom",
        error_kind="tool_error",
        duration_ms=4.0,
    )

    assert success.ok is True
    assert success.content == "8"
    assert success.error_kind is None
    assert success.duration_ms == 12.5

    assert failure.ok is False
    assert failure.content == "Error executing tool 'add': boom"
    assert failure.error_kind == "tool_error"
    assert failure.duration_ms == 4.0


@pytest.mark.asyncio
async def test_event_dispatcher_calls_sync_and_async_handlers():
    seen = []

    def sync_handler(event):
        seen.append(("sync", event.type))

    async def async_handler(event):
        seen.append(("async", event.type))

    dispatcher = EventDispatcher(
        enabled=True,
        handlers=[sync_handler, async_handler],
        log_events=False,
    )

    await dispatcher.emit(RunEvent(type="done", run_id="run_123", content="ok"))

    assert seen == [("sync", "done"), ("async", "done")]


@pytest.mark.asyncio
async def test_event_dispatcher_ignores_events_when_disabled():
    seen = []
    dispatcher = EventDispatcher(
        enabled=False,
        handlers=[lambda event: seen.append(event.type)],
        log_events=False,
    )

    await dispatcher.emit(RunEvent(type="done", run_id="run_123", content="ok"))

    assert seen == []


@pytest.mark.asyncio
async def test_event_dispatcher_logs_json_when_enabled(caplog):
    logger = logging.getLogger("air_agent.tracing")
    dispatcher = EventDispatcher(enabled=True, handlers=[], log_events=True, logger=logger)

    with caplog.at_level(logging.INFO, logger="air_agent.tracing"):
        await dispatcher.emit(RunEvent(type="done", run_id="run_123", content="ok"))

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["type"] == "done"
    assert payload["run_id"] == "run_123"
    assert payload["content"] == "ok"


@pytest.mark.asyncio
async def test_event_dispatcher_handler_exception_does_not_stop_later_handlers(caplog):
    seen = []
    logger = logging.getLogger("air_agent.tracing")

    def bad_handler(event):
        raise RuntimeError("handler failed")

    def later_handler(event):
        seen.append(event.type)

    dispatcher = EventDispatcher(
        enabled=True,
        handlers=[bad_handler, later_handler],
        log_events=False,
        logger=logger,
    )

    with caplog.at_level(logging.WARNING, logger="air_agent.tracing"):
        await dispatcher.emit(RunEvent(type="done", run_id="run_123", content="ok"))

    assert seen == ["done"]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


@pytest.mark.asyncio
async def test_event_dispatcher_logging_serialization_error_warns_and_continues(caplog):
    seen = []
    logger = logging.getLogger("air_agent.tracing")
    dispatcher = EventDispatcher(
        enabled=True,
        handlers=[lambda event: seen.append(event.type)],
        log_events=True,
        logger=logger,
    )
    event = RunEvent(
        type="done",
        run_id="run_123",
        content="ok",
        metadata={"value": object()},
    )

    with caplog.at_level(logging.WARNING, logger="air_agent.tracing"):
        await dispatcher.emit(event)

    assert seen == ["done"]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
