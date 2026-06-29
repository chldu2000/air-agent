import pytest
from air_agent import AgentRole, SubagentAggregation
from air_agent.types import Response, RunEvent, StreamEvent, SubagentResult


def test_response_creation():
    r = Response(content="hello", usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15})
    assert r.content == "hello"
    assert r.usage.total_tokens == 15


def test_stream_event_text():
    e = StreamEvent(type="text", content="hello")
    assert e.type == "text"
    assert e.content == "hello"


def test_stream_event_tool_call():
    e = StreamEvent(type="tool_call", name="read_file", arguments='{"path": "/tmp/a"}')
    assert e.name == "read_file"


def test_stream_event_done():
    e = StreamEvent(type="done", usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15})
    assert e.type == "done"
    assert e.usage.total_tokens == 15


def test_subagent_result_success():
    r = SubagentResult(status="success", content="done")
    assert r.status == "success"
    assert r.content == "done"


def test_subagent_result_timeout():
    r = SubagentResult(status="timeout", content="")
    assert r.status == "timeout"


def test_subagent_result_extended_fields_default_to_empty_values():
    r = SubagentResult(status="success", content="done")

    assert r.role is None
    assert r.task is None
    assert r.events == []
    assert r.metadata == {}


def test_subagent_result_accepts_role_task_events_and_metadata():
    event = RunEvent(type="llm_start", run_id="run_1")
    r = SubagentResult(
        status="success",
        content="done",
        role="reviewer",
        task="Check code",
        events=[event],
        metadata={"duration_ms": 12.5},
    )

    assert r.role == "reviewer"
    assert r.task == "Check code"
    assert r.events == [event]
    assert r.metadata == {"duration_ms": 12.5}


def test_agent_role_defaults_are_serialization_friendly():
    role = AgentRole(name="reviewer")

    assert role.name == "reviewer"
    assert role.description == ""
    assert role.system_prompt is None
    assert role.tools == []
    assert role.skills_dir is None
    assert role.memory_scope is None


def test_subagent_aggregation_literal_export_is_available():
    aggregation: SubagentAggregation = "concat"

    assert aggregation == "concat"
