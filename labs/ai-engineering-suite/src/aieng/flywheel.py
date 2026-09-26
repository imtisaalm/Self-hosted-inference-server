"""Consent-filtered feedback, source-group splits, augmentation, and genuine tiny LoRA."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .core import Store, canonical, digest, number, positive_int
from .guardrails import Guardrails, normalized


@dataclass(frozen=True)
class Example:
    source: str
    prompt: str
    completion: str
    synthetic: bool = False

    def text(self) -> str:
        return self.prompt + "\n" + self.completion + "\n"


class FeedbackStore(Store):
    def __init__(self, path: str = ":memory:"):
        super().__init__(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS feedback("
                        "id TEXT PRIMARY KEY,prompt TEXT,completion TEXT,approved INTEGER)")

    def record(self, identity: str, prompt: str, completion: str, *, consent: bool, approved: bool) -> None:
        if consent is not True:
            raise PermissionError("training consent required; record was not stored")
        if type(approved) is not bool or not identity:
            raise ValueError("explicit approval status and record ID required")
        safe = Guardrails()
        prompt, completion = safe.inbound(prompt), safe.inbound(completion)
        if not prompt.strip() or not completion.strip():
            raise ValueError("nonempty prompt and approved answer required")
        with self.transaction() as db:
            row = db.execute("SELECT prompt,completion,approved FROM feedback WHERE id=?", (identity,)).fetchone()
            if row and tuple(row) != (prompt, completion, int(approved)):
                raise ValueError("feedback records are immutable; use a new ID")
            db.execute("INSERT OR IGNORE INTO feedback VALUES(?,?,?,?)",
                       (identity, prompt, completion, int(approved)))

    def delete(self, identity: str) -> bool:
        with self.transaction() as db:
            return db.execute("DELETE FROM feedback WHERE id=?", (identity,)).rowcount == 1

    def prepare(self, *, holdout_fraction: float = .25,
                augment: Callable[[str], Sequence[str]] | None = None) -> tuple[list[Example], list[Example]]:
        number(holdout_fraction, "holdout_fraction")
        if not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between zero and one")
        with self.lock:
            rows = self.db.execute("SELECT * FROM feedback WHERE approved=1 ORDER BY id").fetchall()
        unique: dict[str, Example] = {}
        for row in rows:
            key = digest(normalized(row['prompt']).casefold())
            example = Example(key, row['prompt'], row['completion'])
            if key in unique and unique[key].completion != example.completion:
                raise ValueError("conflicting approved answers for duplicate prompt")
            unique.setdefault(key, example)
        ordered = sorted(unique.values(), key=lambda row: row.source)
        if len(ordered) < 2:
            raise ValueError("at least two independent approved prompts are required")
        count = min(len(ordered) - 1, max(1, int(len(ordered) * holdout_fraction)))
        holdout, train = ordered[:count], list(ordered[count:])
        # Split originals BEFORE generating variations; reserve all original prompt hashes.
        used = set(unique)
        if augment:
            for original in list(train):
                variants = list(augment(original.prompt))
                if len(variants) > 10:
                    raise ValueError("augmentation cap is ten variants per source")
                for variant in variants:
                    prompt = Guardrails().inbound(variant)
                    key = digest(normalized(prompt).casefold())
                    if not prompt.strip() or key in used:
                        continue
                    used.add(key)
                    train.append(Example(original.source, prompt, original.completion, True))
        return train, holdout


def export_jsonl(path: str | Path, examples: Sequence[Example]) -> None:
    from dataclasses import asdict
    with Path(path).open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(canonical(asdict(example)) + "\n")


class TinyLoRA:
    """Frozen character-bigram LM with trainable low-rank delta: W = W0 + scale*A@B.

    This trains actual weights with analytic gradients. It is a numerical teaching
    model, not a pretrained foundation model or a substitute for foundation-model tuning.
    Vocabulary is fitted to training text only; unseen holdout characters map to UNK.
    """
    def __init__(self, training_texts: Sequence[str], *, rank: int = 4, seed: int = 7):
        self.rank = positive_int(rank, "rank")
        self.vocab = ['<UNK>'] + sorted(set(''.join(training_texts)))
        if len(self.vocab) < 3:
            raise ValueError("training corpus needs at least two characters")
        if len(self.vocab) > 256 or self.rank > len(self.vocab):
            raise ValueError("tiny model supports at most 256 symbols and rank <= vocabulary")
        self.index = {symbol: i for i, symbol in enumerate(self.vocab)}
        self.scale = 1.0 / rank
        rng = np.random.default_rng(seed)
        self.base = rng.normal(0, .01, (len(self.vocab), len(self.vocab)))
        self.base.setflags(write=False)
        self.A = rng.normal(0, .1, (len(self.vocab), rank))
        self.B = np.zeros((rank, len(self.vocab)))

    def pairs(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for text in texts:
            ids = [self.index.get(c, 0) for c in text]
            xs.extend(ids[:-1])
            ys.extend(ids[1:])
        if not xs or len(xs) > 200_000:
            raise ValueError("corpus must contain 1 to 200000 adjacent character pairs")
        return np.asarray(xs), np.asarray(ys)

    def gradients(self, texts: Sequence[str]) -> tuple[float, np.ndarray, np.ndarray]:
        x, y = self.pairs(texts)
        logits = self.base[x] + self.scale * (self.A[x] @ self.B)
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        loss = float(-logp[np.arange(len(y)), y].mean())
        residual = np.exp(logp)
        residual[np.arange(len(y)), y] -= 1
        residual /= len(y)
        gradient = np.zeros_like(self.base)
        np.add.at(gradient, x, residual)
        return loss, self.scale * (gradient @ self.B.T), self.scale * (self.A.T @ gradient)

    def loss(self, texts: Sequence[str]) -> float:
        return self.gradients(texts)[0]

    def fit(self, texts: Sequence[str], *, epochs: int = 250, learning_rate: float = 8) -> list[float]:
        positive_int(epochs, 'epochs')
        number(learning_rate, 'learning_rate', .000001)
        if epochs > 10000:
            raise ValueError('epoch cap exceeded')
        history = []
        for _ in range(epochs):
            loss, da, db = self.gradients(texts)
            norm = float(np.sqrt(np.sum(da * da) + np.sum(db * db)))
            if not np.isfinite(loss) or not np.isfinite(norm):
                raise FloatingPointError('nonfinite optimization state')
            scale = max(1, norm)
            self.A -= learning_rate * da / scale
            self.B -= learning_rate * db / scale
            history.append(loss)
        return history

    def save(self, path: str | Path) -> None:
        metadata = canonical({'vocab': self.vocab, 'rank': self.rank, 'scale': self.scale})
        # No pickle objects in the checkpoint.
        np.savez_compressed(path, base=self.base, A=self.A, B=self.B, metadata=metadata)

    @classmethod
    def load(cls, path: str | Path) -> TinyLoRA:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            model = cls([''.join(meta['vocab'][1:])], rank=meta['rank'])
            if model.vocab != meta['vocab'] or meta['scale'] != model.scale:
                raise ValueError('invalid checkpoint metadata')
            for name in ('base', 'A', 'B'):
                array = archive[name].copy()
                if array.shape != getattr(model, name).shape or not np.isfinite(array).all():
                    raise ValueError('invalid checkpoint weights')
                setattr(model, name, array)
            model.base.setflags(write=False)
        return model


def training_cycle(train: Sequence[Example], holdout: Sequence[Example], *, epochs: int = 250) -> tuple[TinyLoRA, dict[str, Any]]:
    if not train or not holdout or {x.source for x in train} & {x.source for x in holdout}:
        raise ValueError('nonempty, disjoint source-group train/holdout sets required')
    if {digest(normalized(x.prompt).casefold()) for x in train} & \
            {digest(normalized(x.prompt).casefold()) for x in holdout}:
        raise ValueError('prompt leakage across split')
    model = TinyLoRA([x.text() for x in train])
    texts, held = [x.text() for x in train], [x.text() for x in holdout]
    before = model.loss(held)
    history = model.fit(texts, epochs=epochs)
    after = model.loss(held)
    report = {'train_rows': len(train), 'holdout_rows': len(holdout),
              'initial_train_loss': history[0], 'final_train_loss': model.loss(texts),
              'initial_holdout_loss': before, 'final_holdout_loss': after,
              'promotion_eligible': bool(np.isfinite(after) and after < before),
              'metric': 'character_next_token_cross_entropy',
              'model_kind': 'tiny_frozen_bigram_with_lora',
              'dataset_hash': digest([(x.source, x.prompt, x.completion) for x in (*train, *holdout)])}
    return model, report
