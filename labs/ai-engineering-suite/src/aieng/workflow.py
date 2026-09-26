"""SQLite checkpoints with lease fencing; external effects remain at-least-once."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .core import Store, canonical, digest, number


class Busy(RuntimeError):
    """Another worker owns a live lease."""


class LostLease(RuntimeError):
    """A stale worker is prohibited from committing."""


@dataclass(frozen=True)
class Step:
    name: str
    version: str
    run: Callable[[Any, str], Any]


class WorkflowEngine(Store):
    def __init__(self, path: str = ":memory:", *, clock: Callable[[], float] = time.time):
        super().__init__(path)
        self.clock = clock
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS runs(
            id TEXT PRIMARY KEY, plan TEXT NOT NULL, initial TEXT NOT NULL,
            state TEXT NOT NULL, next_step INTEGER NOT NULL, status TEXT NOT NULL,
            owner TEXT, lease REAL NOT NULL, fence INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS checkpoints(
            run_id TEXT, step INTEGER, state TEXT NOT NULL, idempotency_key TEXT NOT NULL,
            PRIMARY KEY(run_id,step));
        """)

    def run(self, run_id: str, initial: Any, steps: Sequence[Step], *, lease_s: float = 30) -> Any:
        duration = number(lease_s, "lease_s")
        if not duration or not run_id or not steps:
            raise ValueError("run ID, steps, and positive lease required")
        if any(not s.name or not s.version for s in steps):
            raise ValueError("steps require explicit names and versions")
        if len({s.name for s in steps}) != len(steps):
            raise ValueError("step names must be unique")
        plan = digest([(s.name, s.version) for s in steps])
        first = canonical(initial)
        owner = uuid.uuid4().hex
        with self.transaction() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                db.execute("INSERT INTO runs VALUES(?,?,?,?,0,'pending',NULL,0,0)",
                           (run_id, plan, first, first))
                row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row["plan"] != plan or row["initial"] != first:
                raise ValueError("run ID reused with different inputs or step versions")
            if row["status"] == "complete":
                return json.loads(row["state"])
            if row["owner"] and row["lease"] > self.clock():
                raise Busy(run_id)
            fence = row["fence"] + 1
            db.execute("UPDATE runs SET owner=?,lease=?,fence=?,status='running' WHERE id=?",
                       (owner, self.clock() + duration, fence, run_id))
            position, state = row["next_step"], json.loads(row["state"])
        try:
            for index in range(position, len(steps)):
                step = steps[index]
                key = digest((run_id, plan, index))
                # No database transaction spans an external side effect.
                result = step.run(state, key)
                encoded = canonical(result)
                now = self.clock()
                with self.transaction() as db:
                    changed = db.execute(
                        "UPDATE runs SET state=?,next_step=?,lease=? WHERE id=? AND owner=? "
                        "AND fence=? AND lease>?",
                        (encoded, index + 1, now + duration, run_id, owner, fence, now),
                    ).rowcount
                    if changed != 1:
                        raise LostLease(run_id)
                    db.execute("INSERT INTO checkpoints VALUES(?,?,?,?)",
                               (run_id, index, encoded, key))
                state = json.loads(encoded)
            with self.transaction() as db:
                changed = db.execute(
                    "UPDATE runs SET status='complete',owner=NULL,lease=0 "
                    "WHERE id=? AND owner=? AND fence=? AND lease>?",
                    (run_id, owner, fence, self.clock()),
                ).rowcount
                if changed != 1:
                    raise LostLease(run_id)
            return state
        except BaseException:
            with self.transaction() as db:
                db.execute("UPDATE runs SET status='pending',owner=NULL,lease=0 "
                           "WHERE id=? AND owner=? AND fence=?", (run_id, owner, fence))
            raise

    def inspect(self, run_id: str) -> dict[str, Any]:
        with self.lock:
            row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            return {"id": run_id, "next_step": row["next_step"], "status": row["status"],
                    "state": json.loads(row["state"]), "fence": row["fence"]}
