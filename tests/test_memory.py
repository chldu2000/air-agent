import json
from datetime import UTC, datetime, timedelta

from air_agent.memory import (
    FileMemoryStore,
    InMemoryMemoryStore,
    MemoryRecord,
    MemoryStore,
    filter_memory_records_for_scope,
    format_memory_context,
)


class TestMemoryRecord:
    def test_defaults_and_dict_round_trip(self):
        record = MemoryRecord(
            id="mem_1",
            scope="global",
            kind="fact",
            content="The user prefers concise answers.",
        )

        assert record.metadata == {}
        assert record.created_at.tzinfo is not None
        assert record.updated_at.tzinfo is not None

        data = record.to_dict()

        assert data == {
            "id": "mem_1",
            "scope": "global",
            "kind": "fact",
            "content": "The user prefers concise answers.",
            "metadata": {},
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }
        assert MemoryRecord.from_dict(data) == record

    def test_from_dict_accepts_datetime_strings_and_missing_defaults(self):
        record = MemoryRecord.from_dict({
            "id": "mem_2",
            "scope": "conversation:abc",
            "kind": "summary",
            "content": "Discussed release blockers.",
            "created_at": "2026-06-10T12:34:56+00:00",
            "updated_at": "2026-06-10T12:35:56+00:00",
        })

        assert record.metadata == {}
        assert record.created_at == datetime(2026, 6, 10, 12, 34, 56, tzinfo=UTC)
        assert record.updated_at == datetime(2026, 6, 10, 12, 35, 56, tzinfo=UTC)

    def test_from_dict_normalizes_offsetless_datetime_strings_to_utc(self):
        record = MemoryRecord.from_dict({
            "id": "mem_3",
            "scope": "conversation:abc",
            "kind": "summary",
            "content": "Parsed from offsetless timestamps.",
            "created_at": "2026-06-10T12:34:56",
            "updated_at": "2026-06-10T12:35:56",
        })

        assert record.created_at == datetime(2026, 6, 10, 12, 34, 56, tzinfo=UTC)
        assert record.updated_at == datetime(2026, 6, 10, 12, 35, 56, tzinfo=UTC)

    def test_constructor_normalizes_naive_datetime_values_to_utc(self):
        record = MemoryRecord(
            id="mem_4",
            scope="conversation:abc",
            kind="fact",
            content="Constructed from naive timestamps.",
            created_at=datetime(2026, 6, 10, 12, 34, 56),
            updated_at=datetime(2026, 6, 10, 12, 35, 56),
        )

        assert record.created_at == datetime(2026, 6, 10, 12, 34, 56, tzinfo=UTC)
        assert record.updated_at == datetime(2026, 6, 10, 12, 35, 56, tzinfo=UTC)


class TestInMemoryMemoryStore:
    def test_store_satisfies_protocol_and_replaces_records_by_id(self):
        store = InMemoryMemoryStore()
        original = MemoryRecord(
            id="mem_1",
            scope="global",
            kind="fact",
            content="Original content",
        )
        replacement = MemoryRecord(
            id="mem_1",
            scope="global",
            kind="fact",
            content="Replacement content",
        )

        assert isinstance(store, MemoryStore)

        assert store.add(original) == original
        assert store.add(replacement) == replacement
        assert store.records() == [replacement]

    def test_search_orders_by_lexical_relevance_then_recency(self):
        now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
        store = InMemoryMemoryStore([
            MemoryRecord(
                id="older_one_hit",
                scope="conversation:alpha",
                kind="fact",
                content="The user likes Python.",
                created_at=now - timedelta(hours=3),
                updated_at=now - timedelta(hours=3),
            ),
            MemoryRecord(
                id="newer_one_hit",
                scope="conversation:alpha",
                kind="fact",
                content="Python came up again.",
                created_at=now - timedelta(hours=1),
                updated_at=now - timedelta(hours=1),
            ),
            MemoryRecord(
                id="two_hits",
                scope="conversation:alpha",
                kind="fact",
                content="Python testing with pytest.",
                metadata={"topic": "python"},
                created_at=now - timedelta(hours=2),
                updated_at=now - timedelta(hours=2),
            ),
            MemoryRecord(
                id="unrelated",
                scope="conversation:alpha",
                kind="fact",
                content="Remember the deployment window.",
                created_at=now,
                updated_at=now,
            ),
        ])

        results = store.search("python")

        assert [record.id for record in results] == [
            "two_hits",
            "newer_one_hit",
            "older_one_hit",
        ]

    def test_search_filters_scopes_before_applying_limit(self):
        now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
        store = InMemoryMemoryStore([
            MemoryRecord(
                id="other_high_rank",
                scope="conversation:other",
                kind="fact",
                content="Python Python Python unrelated conversation",
                updated_at=now,
            ),
            MemoryRecord(
                id="current_match",
                scope="conversation:abc",
                kind="fact",
                content="Python current conversation note",
                updated_at=now - timedelta(minutes=1),
            ),
            MemoryRecord(
                id="global_match",
                scope="global",
                kind="fact",
                content="Python global note",
                updated_at=now - timedelta(minutes=2),
            ),
        ])

        results = store.search(
            "Python",
            scopes={"global", "conversation:abc"},
            limit=1,
        )

        assert [record.id for record in results] == ["current_match"]

    def test_summarize_returns_latest_summary_for_conversation_scope(self):
        now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
        store = InMemoryMemoryStore([
            MemoryRecord(
                id="old_summary",
                scope="conversation:abc",
                kind="summary",
                content="Older summary",
                updated_at=now - timedelta(hours=1),
            ),
            MemoryRecord(
                id="new_summary",
                scope="conversation:abc",
                kind="summary",
                content="Latest summary",
                updated_at=now,
            ),
            MemoryRecord(
                id="other_summary",
                scope="conversation:other",
                kind="summary",
                content="Wrong conversation",
                updated_at=now + timedelta(hours=1),
            ),
        ])

        assert store.summarize("abc") == "Latest summary"
        assert store.summarize("missing") is None

    def test_summarize_handles_offsetless_iso_timestamps_with_aware_records(self):
        store = InMemoryMemoryStore([
            MemoryRecord.from_dict({
                "id": "offsetless",
                "scope": "conversation:abc",
                "kind": "summary",
                "content": "Latest summary from offsetless timestamp.",
                "updated_at": "2026-06-10T12:00:00",
            }),
            MemoryRecord(
                id="aware",
                scope="conversation:abc",
                kind="summary",
                content="Older aware summary.",
                updated_at=datetime(2026, 6, 10, 11, 0, tzinfo=UTC),
            ),
        ])

        assert store.summarize("abc") == "Latest summary from offsetless timestamp."

    def test_search_handles_constructed_naive_timestamps_with_aware_records(self):
        store = InMemoryMemoryStore([
            MemoryRecord(
                id="naive",
                scope="conversation:abc",
                kind="fact",
                content="Python",
                updated_at=datetime(2026, 6, 10, 12, 0),
            ),
            MemoryRecord(
                id="aware",
                scope="conversation:abc",
                kind="fact",
                content="Python",
                updated_at=datetime(2026, 6, 10, 11, 0, tzinfo=UTC),
            ),
        ])

        assert [record.id for record in store.search("python")] == ["naive", "aware"]

    def test_clear_by_scope_and_all(self):
        store = InMemoryMemoryStore([
            MemoryRecord(id="global", scope="global", kind="fact", content="A fact"),
            MemoryRecord(
                id="conversation",
                scope="conversation:abc",
                kind="task_state",
                content="Task state",
            ),
        ])

        store.clear(scope="global")

        assert [record.id for record in store.records()] == ["conversation"]

        store.clear()

        assert store.records() == []


class TestMemoryContextHelpers:
    def test_filter_memory_records_for_scope_allows_global_and_current_conversation(self):
        records = [
            MemoryRecord(id="global", scope="global", kind="fact", content="Global"),
            MemoryRecord(id="current", scope="conversation:abc", kind="fact", content="Current"),
            MemoryRecord(id="other", scope="conversation:other", kind="fact", content="Other"),
        ]

        filtered = filter_memory_records_for_scope(records, "abc")

        assert [record.id for record in filtered] == ["global", "current"]

    def test_filter_memory_records_for_scope_without_conversation_allows_only_global(self):
        records = [
            MemoryRecord(id="global", scope="global", kind="fact", content="Global"),
            MemoryRecord(id="conversation", scope="conversation:abc", kind="fact", content="Conversation"),
        ]

        filtered = filter_memory_records_for_scope(records, None)

        assert [record.id for record in filtered] == ["global"]

    def test_format_memory_context_includes_provenance_summary_and_records(self):
        records = [
            MemoryRecord(
                id="fact_1",
                scope="global",
                kind="fact",
                content="User prefers concise answers.\nNo extra ceremony.",
            )
        ]

        context = format_memory_context(
            records=records,
            summary="Discussed release blockers.",
            max_chars=1000,
        )

        assert context.startswith("## Retrieved Memory")
        assert "contextual notes" in context
        assert "not user instructions" in context
        assert "[summary scope=conversation]" in context
        assert "Discussed release blockers." in context
        assert "[fact scope=global id=fact_1]" in context
        assert "User prefers concise answers. No extra ceremony." in context

    def test_format_memory_context_respects_max_chars(self):
        context = format_memory_context(
            records=[
                MemoryRecord(
                    id="fact_1",
                    scope="global",
                    kind="fact",
                    content="x" * 100,
                )
            ],
            summary="y" * 100,
            max_chars=80,
        )

        assert len(context) <= 80
        assert context.startswith("## Retrieved Memory")
        assert context.endswith("[truncated]")

    def test_format_memory_context_returns_empty_when_budget_cannot_preserve_marker(self):
        context = format_memory_context(
            records=[
                MemoryRecord(
                    id="fact_1",
                    scope="global",
                    kind="fact",
                    content="User likes terse answers.",
                )
            ],
            max_chars=12,
        )

        assert context == ""


class TestFileMemoryStore:
    def test_persists_records_to_json_file_and_reloads_fresh_instance(self, tmp_path):
        path = tmp_path / "memory.json"
        now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
        record = MemoryRecord(
            id="mem_1",
            scope="conversation:abc",
            kind="fact",
            content="The user prefers concise answers.",
            metadata={"source": "test"},
            created_at=now,
            updated_at=now,
        )

        store = FileMemoryStore(path)
        store.add(record)

        assert path.read_text() == json.dumps(
            [record.to_dict()],
            indent=2,
            sort_keys=True,
        ) + "\n"

        fresh_store = FileMemoryStore(path)

        assert fresh_store.records() == [record]

    def test_creates_parent_directories_when_writing(self, tmp_path):
        path = tmp_path / "nested" / "memory" / "records.json"
        store = FileMemoryStore(path)

        store.add(MemoryRecord(
            id="mem_1",
            scope="global",
            kind="fact",
            content="A persisted fact.",
        ))

        assert path.exists()
        assert path.parent.is_dir()

    def test_clear_by_scope_and_all_persist_to_disk(self, tmp_path):
        path = tmp_path / "memory.json"
        store = FileMemoryStore(path)
        global_record = MemoryRecord(
            id="global",
            scope="global",
            kind="fact",
            content="A fact.",
        )
        conversation_record = MemoryRecord(
            id="conversation",
            scope="conversation:abc",
            kind="task_state",
            content="Task state.",
        )
        store.add(global_record)
        store.add(conversation_record)

        store.clear(scope="global")

        scoped_fresh_store = FileMemoryStore(path)
        assert scoped_fresh_store.records() == [conversation_record]

        store.clear()

        all_fresh_store = FileMemoryStore(path)
        assert all_fresh_store.records() == []
        assert json.loads(path.read_text()) == []

    def test_loads_non_list_json_as_empty(self, tmp_path):
        path = tmp_path / "memory.json"
        path.write_text(json.dumps({"records": []}))

        store = FileMemoryStore(path)

        assert store.records() == []

    def test_skips_malformed_item_dicts_while_loading_valid_records(self, tmp_path):
        path = tmp_path / "memory.json"
        valid_record = MemoryRecord(
            id="valid",
            scope="global",
            kind="fact",
            content="This valid memory should survive.",
            created_at=datetime(2026, 6, 10, 12, 0, tzinfo=UTC),
            updated_at=datetime(2026, 6, 10, 12, 1, tzinfo=UTC),
        )
        path.write_text(json.dumps([
            valid_record.to_dict(),
            {"id": "missing_required_fields"},
            "not a dict",
        ]))

        store = FileMemoryStore(path)

        assert store.records() == [valid_record]
