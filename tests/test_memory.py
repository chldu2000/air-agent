from datetime import UTC, datetime, timedelta

from air_agent.memory import InMemoryMemoryStore, MemoryRecord, MemoryStore


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
