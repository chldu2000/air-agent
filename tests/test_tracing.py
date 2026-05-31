from __future__ import annotations

from datetime import datetime, timezone

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
