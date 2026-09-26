"""SQLite cosine cache with strict namespaces, TTL, capacity, and hit counters."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .core import Store, canonical, digest, number, positive_int
from .retrieval import cosine, unit


@dataclass(frozen=True)
class CacheScope:
    tenant: str
    model: str
    prompt_version: str
    embedding_model: str
    corpus_version: str

    def key(self) -> str:
        fields = (self.tenant, self.model, self.prompt_version, self.embedding_model, self.corpus_version)
        if not all(isinstance(field, str) and field for field in fields):
            raise ValueError("all cache scope fields must be nonempty")
        return digest(fields)


@dataclass(frozen=True)
class CacheHit:
    value: Any
    similarity: float
    exact: bool


class SemanticCache(Store):
    def __init__(self, path: str | Path = ":memory:", capacity: int = 1000, clock: Callable[[], float] = time.time) -> None:
        positive_int(capacity, "capacity")
        super().__init__(path)
        self.capacity, self.clock = capacity, clock
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS cache(
            scope TEXT, key TEXT, vector TEXT, value TEXT, expires REAL, touched REAL,
            PRIMARY KEY(scope,key));
          CREATE TABLE IF NOT EXISTS cache_stats(scope TEXT PRIMARY KEY,hits INTEGER,misses INTEGER);
        """)

    def put(self, scope: CacheScope, key: str, vector: Sequence[float], value: Any, ttl: float = 300) -> None:
        number(ttl, "ttl", 0.000001)
        ns, embedded, payload = scope.key(), canonical(unit(vector)), canonical(value)
        if not isinstance(key, str) or not key:
            raise ValueError("cache key must be nonempty")
        # Store a hash rather than the original question in the key column.
        key_hash, now = digest(key), self.clock()
        with self.transaction() as db:
            db.execute("DELETE FROM cache WHERE expires <= ?", (now,))
            old = db.execute("SELECT vector FROM cache WHERE scope=? LIMIT 1", (ns,)).fetchone()
            if old and len(json.loads(old[0])) != len(json.loads(embedded)):
                raise ValueError("embedding dimensions changed inside one scope")
            db.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?,?,?,?)", (ns,key_hash,embedded,payload,now+ttl,now))
            db.execute("DELETE FROM cache WHERE rowid IN (SELECT rowid FROM cache ORDER BY touched DESC,rowid DESC LIMIT -1 OFFSET ?)", (self.capacity,))

    def get(self, scope: CacheScope, key: str, vector: Sequence[float], threshold: float = 0.95) -> CacheHit | None:
        if not 0 <= number(threshold, "threshold") <= 1:
            raise ValueError("threshold must be <= 1")
        ns, query, now, key_hash = scope.key(), unit(vector), self.clock(), digest(key)
        with self.transaction() as db:
            db.execute("DELETE FROM cache WHERE expires <= ?", (now,))
            rows = db.execute("SELECT * FROM cache WHERE scope=? ORDER BY key", (ns,)).fetchall()
            scored = [(cosine(query, json.loads(row["vector"])), row) for row in rows]
            exact = next(((score,row) for score,row in scored if row["key"] == key_hash), None)
            best = exact or max(scored, key=lambda pair: pair[0], default=None)
            hit = best is not None and (exact is not None or best[0] >= threshold)
            db.execute("INSERT INTO cache_stats VALUES(?,?,?) ON CONFLICT(scope) DO UPDATE SET hits=hits+excluded.hits,misses=misses+excluded.misses", (ns,int(hit),int(not hit)))
            if hit and best:
                score, row = best
                db.execute("UPDATE cache SET touched=? WHERE scope=? AND key=?", (now, ns, row["key"]))
                return CacheHit(json.loads(row["value"]), score, exact is not None)
        return None

    def stats(self, scope: CacheScope) -> dict[str, float | int]:
        with self.lock:
            row = self.db.execute("SELECT hits,misses FROM cache_stats WHERE scope=?", (scope.key(),)).fetchone()
        hits, misses = tuple(row) if row else (0, 0)
        return {"hits": hits, "misses": misses, "hit_rate": hits / (hits + misses) if hits + misses else 0.0}

    def invalidate(self, scope: CacheScope) -> int:
        with self.transaction() as db:
            return db.execute("DELETE FROM cache WHERE scope=?", (scope.key(),)).rowcount
