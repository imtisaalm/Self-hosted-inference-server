"""Immutable prompt versions, transactional deployments, stable weighted experiments."""
from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .core import Store, canonical, digest, positive_int


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    template: str
    hash: str

    def render(self, **variables: str) -> str:
        template = string.Template(self.template)
        if set(variables) != set(template.get_identifiers()):
            raise ValueError("template variables must match exactly")
        return template.substitute(variables)


class PromptRegistry(Store):
    def __init__(self, path: str | Path = ":memory:") -> None:
        super().__init__(path)
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS prompts(name TEXT,version INTEGER,template TEXT,hash TEXT,
            PRIMARY KEY(name,version),UNIQUE(name,hash));
          CREATE TABLE IF NOT EXISTS deployments(name TEXT PRIMARY KEY,version INTEGER);
          CREATE TABLE IF NOT EXISTS history(seq INTEGER PRIMARY KEY,name TEXT,old INTEGER,new INTEGER);
          CREATE TABLE IF NOT EXISTS experiments(name TEXT,experiment TEXT,weights TEXT,
            PRIMARY KEY(name,experiment));
        """)

    def register(self, name: str, template: str) -> Prompt:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", name):
            raise ValueError("invalid prompt name")
        if not template or not string.Template(template).is_valid():
            raise ValueError("template must be nonempty with valid $placeholders")
        content_hash = digest(template)
        with self.transaction() as db:
            existing = db.execute("SELECT * FROM prompts WHERE name=? AND hash=?", (name, content_hash)).fetchone()
            if existing:
                return Prompt(*tuple(existing))
            version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM prompts WHERE name=?", (name,)).fetchone()[0]
            db.execute("INSERT INTO prompts VALUES(?,?,?,?)", (name,version,template,content_hash))
        return Prompt(name, version, template, content_hash)

    def get(self, name: str, version: int | None = None) -> Prompt:
        with self.lock:
            if version is None:
                row = self.db.execute("SELECT version FROM deployments WHERE name=?", (name,)).fetchone()
                if not row:
                    raise KeyError("prompt has no active deployment")
                version = row[0]
            row = self.db.execute("SELECT * FROM prompts WHERE name=? AND version=?", (name,version)).fetchone()
        if not row:
            raise KeyError("unknown prompt version")
        return Prompt(*tuple(row))

    def promote(self, name: str, version: int) -> None:
        self.get(name, version)
        with self.transaction() as db:
            row = db.execute("SELECT version FROM deployments WHERE name=?", (name,)).fetchone()
            old = row[0] if row else None
            if old == version:
                return
            db.execute("INSERT INTO history(name,old,new) VALUES(?,?,?)", (name,old,version))
            db.execute("INSERT INTO deployments VALUES(?,?) ON CONFLICT(name) DO UPDATE SET version=excluded.version", (name,version))

    def rollback(self, name: str) -> Prompt:
        with self.transaction() as db:
            row = db.execute("SELECT old,new FROM history WHERE name=? ORDER BY seq DESC LIMIT 1", (name,)).fetchone()
            if not row or row[0] is None:
                raise ValueError("no previous deployment")
            db.execute("UPDATE deployments SET version=? WHERE name=?", (row[0],name))
            db.execute("INSERT INTO history(name,old,new) VALUES(?,?,?)", (name,row[1],row[0]))
            version = row[0]
        return self.get(name, version)

    def experiment(self, name: str, experiment: str, weights: Mapping[int, int]) -> None:
        if not experiment or not weights:
            raise ValueError("experiment and weights required")
        for version, weight in weights.items():
            positive_int(weight, "weight")
            self.get(name, version)
        # An experiment is immutable: changing weights requires a new experiment ID.
        payload = canonical({str(version): weight for version, weight in weights.items()})
        with self.transaction() as db:
            old = db.execute("SELECT weights FROM experiments WHERE name=? AND experiment=?", (name,experiment)).fetchone()
            if old and old[0] != payload:
                raise ValueError("experiment already exists with different weights")
            db.execute("INSERT OR IGNORE INTO experiments VALUES(?,?,?)", (name,experiment,payload))

    def assign(self, name: str, experiment: str, subject: str) -> Prompt:
        if not subject:
            raise ValueError("stable subject identifier required")
        with self.lock:
            row = self.db.execute("SELECT weights FROM experiments WHERE name=? AND experiment=?", (name,experiment)).fetchone()
        if not row:
            raise KeyError("unknown experiment")
        weights = sorted((int(version),weight) for version,weight in json.loads(row[0]).items())
        bucket = int(digest((name,experiment,subject)),16) % sum(weight for _,weight in weights)
        for version, weight in weights:
            if bucket < weight:
                return self.get(name,version)
            bucket -= weight
        raise AssertionError("unreachable assignment")
