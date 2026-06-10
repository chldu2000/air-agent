from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable


MemoryKind = Literal["summary", "fact", "task_state"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_datetime(value: datetime | str | None) -> datetime:
    if value is None:
        return _utc_now()
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


@dataclass
class MemoryRecord:
    id: str
    scope: str
    kind: MemoryKind
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "kind": self.kind,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        return cls(
            id=data["id"],
            scope=data["scope"],
            kind=data["kind"],
            content=data["content"],
            metadata=dict(data.get("metadata") or {}),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


@runtime_checkable
class MemoryStore(Protocol):
    def add(self, record: MemoryRecord) -> MemoryRecord:
        ...

    def search(
        self,
        query: str,
        *,
        scope: str | None = None,
        kind: MemoryKind | None = None,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        ...

    def summarize(self, conversation_id: str) -> str | None:
        ...

    def clear(self, *, scope: str | None = None) -> None:
        ...


class InMemoryMemoryStore:
    def __init__(self, records: list[MemoryRecord] | None = None):
        self._records: dict[str, MemoryRecord] = {}
        for record in records or []:
            self.add(record)

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.id] = record
        return record

    def search(
        self,
        query: str,
        *,
        scope: str | None = None,
        kind: MemoryKind | None = None,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        scored: list[tuple[int, datetime, MemoryRecord]] = []
        for record in self._records.values():
            if scope is not None and record.scope != scope:
                continue
            if kind is not None and record.kind != kind:
                continue
            score = self._score(record, query)
            if score > 0 or not query.strip():
                scored.append((score, record.updated_at, record))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        results = [record for _, _, record in scored]
        if limit is not None:
            return results[:limit]
        return results

    def summarize(self, conversation_id: str) -> str | None:
        scope = f"conversation:{conversation_id}"
        summaries = [
            record
            for record in self._records.values()
            if record.scope == scope and record.kind == "summary"
        ]
        if not summaries:
            return None
        return max(summaries, key=lambda record: record.updated_at).content

    def clear(self, *, scope: str | None = None) -> None:
        if scope is None:
            self._records.clear()
            return
        self._records = {
            record_id: record
            for record_id, record in self._records.items()
            if record.scope != scope
        }

    def records(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def _score(self, record: MemoryRecord, query: str) -> int:
        terms = [term for term in query.lower().split() if term]
        if not terms:
            return 0
        haystack = " ".join(
            [
                record.scope,
                record.kind,
                record.content,
                _metadata_text(record.metadata),
            ]
        ).lower()
        return sum(haystack.count(term) for term in terms)


def _metadata_text(metadata: dict[str, Any]) -> str:
    return " ".join(f"{key} {value}" for key, value in metadata.items())


__all__ = [
    "InMemoryMemoryStore",
    "MemoryKind",
    "MemoryRecord",
    "MemoryStore",
]
