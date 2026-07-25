"""Local, consent-gated episodic memory. No customer content leaves SQLite."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Protocol
from uuid import uuid4


class PayloadCipher(Protocol):
    """Inject a SQLCipher/KMS-backed implementation in production."""
    key_id: str
    def encrypt(self, plaintext: str) -> str: ...
    def decrypt(self, ciphertext: str) -> str: ...


class PlaintextCipher:
    """Development-only adapter. It makes the encryption boundary explicit."""
    key_id = "development-plaintext"
    def encrypt(self, plaintext: str) -> str:
        return plaintext
    def decrypt(self, ciphertext: str) -> str:
        return ciphertext


@dataclass(frozen=True)
class MemoryEntry:
    user_id: str
    interaction_type: str
    content: dict[str, Any]
    context: dict[str, Any]
    outcome: dict[str, Any]
    consented: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    emotional_state: str | None = None
    products_mentioned: list[str] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.astimezone(timezone.utc).isoformat()
        return data


@dataclass(frozen=True)
class PersonalizationProfile:
    user_id: str
    preferred_contact_method: str | None
    product_interest_counts: dict[str, int]
    common_concerns: list[str]
    satisfaction_scores: list[float]
    memory_count: int


class MemoryStore(ABC):
    @abstractmethod
    def save(self, entry: MemoryEntry) -> str: ...
    @abstractmethod
    def search(self, user_id: str, query: str, limit: int = 10) -> list[MemoryEntry]: ...
    @abstractmethod
    def forget_user(self, user_id: str) -> int: ...


class SQLiteMemoryStore(MemoryStore):
    """A deterministic local adapter, replaceable by a pgvector/Qdrant adapter."""
    def __init__(self, database_path: str | Path, cipher: PayloadCipher | None = None):
        self.database_path = str(database_path)
        self.cipher = cipher or PlaintextCipher()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    consented INTEGER NOT NULL,
                    payload_ciphertext TEXT NOT NULL,
                    cipher_key_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user_time ON memories(user_id, timestamp DESC);
                CREATE TABLE IF NOT EXISTS memory_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    subject_hash TEXT NOT NULL,
                    detail TEXT NOT NULL
                );
            """)

    def _audit(self, conn: sqlite3.Connection, event_type: str, user_id: str, detail: dict[str, Any]) -> None:
        subject_hash = sha256(user_id.encode()).hexdigest()
        conn.execute("INSERT INTO memory_audit(timestamp, event_type, subject_hash, detail) VALUES (?, ?, ?, ?)",
                     (datetime.now(timezone.utc).isoformat(), event_type, subject_hash, json.dumps(detail, sort_keys=True)))

    def save(self, entry: MemoryEntry) -> str:
        if not entry.consented:
            raise PermissionError("Episodic memory requires explicit consent.")
        plaintext = json.dumps(entry.payload(), sort_keys=True, default=str)
        with self._connect() as conn:
            conn.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (entry.memory_id, entry.user_id, entry.timestamp.astimezone(timezone.utc).isoformat(),
                          entry.interaction_type, 1, self.cipher.encrypt(plaintext), self.cipher.key_id))
            self._audit(conn, "memory_saved", entry.user_id, {"memory_id": entry.memory_id, "interaction_type": entry.interaction_type})
        return entry.memory_id

    def _deserialize(self, row: sqlite3.Row) -> MemoryEntry:
        payload = json.loads(self.cipher.decrypt(row["payload_ciphertext"]))
        payload["timestamp"] = datetime.fromisoformat(payload["timestamp"])
        return MemoryEntry(**payload)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9_]+", value.lower()) if len(token) > 2}

    def search(self, user_id: str, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Customer-scoped, stable keyword + recency ranking; ties break by ID."""
        query_tokens = self._tokens(query)
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM memories WHERE user_id = ? AND consented = 1", (user_id,)).fetchall()
            self._audit(conn, "memory_retrieved", user_id, {"query_token_count": len(query_tokens), "candidate_count": len(rows)})
        scored: list[tuple[float, str, MemoryEntry]] = []
        now = datetime.now(timezone.utc)
        for row in rows:
            entry = self._deserialize(row)
            searchable = json.dumps({"content": entry.content, "context": entry.context, "products": entry.products_mentioned}, sort_keys=True)
            overlap = len(query_tokens & self._tokens(searchable))
            age_days = max((now - entry.timestamp.astimezone(timezone.utc)).total_seconds() / 86400, 0)
            score = (10 * overlap) + (1 / (1 + age_days))
            scored.append((score, entry.memory_id, entry))
        return [entry for _, _, entry in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]

    def build_profile(self, user_id: str, limit: int = 100) -> PersonalizationProfile:
        memories = self.search(user_id, "", limit=limit)
        contacts = [m.context.get("contact_method") for m in memories if m.context.get("contact_method")]
        products = Counter(product for m in memories for product in m.products_mentioned)
        concerns = Counter(str(m.content.get("topic", "")) for m in memories if m.content.get("topic"))
        scores = [float(m.outcome["satisfaction_score"]) for m in memories if m.outcome.get("satisfaction_score") is not None]
        preferred = sorted(Counter(contacts).items(), key=lambda pair: (-pair[1], pair[0]))[0][0] if contacts else None
        return PersonalizationProfile(user_id, preferred, dict(products), [name for name, _ in concerns.most_common(5)], scores, len(memories))

    def forget_user(self, user_id: str) -> int:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)).fetchone()[0]
            conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            # The audit is de-identified and records only the deletion operation.
            self._audit(conn, "user_forgotten", user_id, {"deleted_memory_count": count})
        return int(count)


class MemoryAwareRecommendation:
    """Applies a deterministic, non-eligibility-overriding ranking adjustment."""
    @staticmethod
    def adjust(recommendations: list[dict[str, Any]], profile: PersonalizationProfile) -> list[dict[str, Any]]:
        adjusted = []
        for recommendation in recommendations:
            item = dict(recommendation)
            product = str(item.get("product", ""))
            interest = profile.product_interest_counts.get(product, 0)
            item["memory_interest_count"] = interest
            item["memory_rationale"] = "Prior interest recorded" if interest else "No relevant consented memory"
            # Eligibility is never changed; score adjustment is deterministic and bounded.
            item["memory_rank_adjustment"] = min(interest, 3) if item.get("eligible", True) else 0
            adjusted.append(item)
        return sorted(adjusted, key=lambda item: (-item["memory_rank_adjustment"], str(item.get("product", ""))))
