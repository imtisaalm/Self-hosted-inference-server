from dataclasses import replace

import numpy as np
import pytest

from aieng.evals import Case, Score, evaluate, gate, smoke_cases, trajectory_grade
from aieng.flywheel import FeedbackStore, TinyLoRA, training_cycle


def dataset():
    store = FeedbackStore()
    for i, prompt in enumerate(['what is a cache', 'what is a router', 'how does retrieval work',
                                'explain model latency', 'explain memory', 'what is an agent']):
        store.record(str(i), prompt, 'a system uses data to answer a question', consent=True, approved=True)
    return store


def test_feedback_consent_approval_redaction_dedup_and_split():
    with dataset() as store:
        with pytest.raises(PermissionError):
            store.record('no', 'private', 'private', consent=False, approved=True)
        store.record('pending', 'pending answer', 'x', consent=True, approved=False)
        store.record('duplicate', 'WHAT  is a cache', 'a system uses data to answer a question', consent=True, approved=True)
        train, holdout = store.prepare(augment=lambda p: [p, 'Question: ' + p])
        assert len({x.source for x in train}) + len(holdout) == 6
        assert {x.source for x in train}.isdisjoint(x.source for x in holdout)
        assert any(x.synthetic for x in train) and not any(x.synthetic for x in holdout)
        assert not any(x.prompt == 'pending answer' for x in train + holdout)
        assert store.delete('pending') and not store.delete('pending')


def test_feedback_conflicts_and_immutability():
    with FeedbackStore() as s:
        s.record('one', 'question', 'answer', consent=True, approved=True)
        with pytest.raises(ValueError):
            s.record('one', 'question', 'different', consent=True, approved=True)
        s.record('two', 'QUESTION', 'contradiction', consent=True, approved=True)
        with pytest.raises(ValueError):
            s.prepare()


def test_lora_gradient_finite_difference():
    texts = ['ababa', 'babaa']
    model = TinyLoRA(texts, rank=2)
    model.B[:] = np.random.default_rng(10).normal(0, .05, model.B.shape)
    _, da, db = model.gradients(texts)
    epsilon = 1e-6
    for matrix, grad in [(model.A, da), (model.B, db)]:
        for index in [(0, 0), (1, 1)]:
            original = matrix[index]
            matrix[index] = original + epsilon
            plus = model.loss(texts)
            matrix[index] = original - epsilon
            minus = model.loss(texts)
            matrix[index] = original
            assert grad[index] == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-7)


def test_lora_frozen_base_loss_and_checkpoint(tmp_path):
    texts = ['abababab', 'babababa']
    model = TinyLoRA(texts, rank=2)
    base = model.base.copy()
    initial = model.loss(texts)
    model.fit(texts, epochs=120)
    assert model.loss(texts) < initial
    np.testing.assert_array_equal(base, model.base)
    with pytest.raises(ValueError):
        model.base[0, 0] = 99
    path = tmp_path / 'adapter.npz'
    model.save(path)
    loaded = TinyLoRA.load(path)
    assert loaded.loss(texts) == pytest.approx(model.loss(texts))


def test_training_cycle_holdout_and_leakage():
    with dataset() as s:
        train, holdout = s.prepare(augment=lambda p: ['Question: ' + p])
    model, report = training_cycle(train, holdout, epochs=120)
    assert report['final_train_loss'] < report['initial_train_loss']
    assert np.isfinite(report['final_holdout_loss'])
    assert report['promotion_eligible'] == (report['final_holdout_loss'] < report['initial_holdout_loss'])
    with pytest.raises(ValueError):
        training_cycle(train, [train[0]])


def test_eval_actual_cases_and_gate():
    scores = evaluate(smoke_cases())
    assert all(s.score == 1 for s in scores)
    assert gate(scores, scores)[0]
    regressed = [replace(scores[0], score=0, critical=False)] + scores[1:]
    passed, reasons = gate(regressed, scores)
    assert not passed and any('critical' in reason for reason in reasons)
    assert not gate(scores[:-1], scores)[0]


def test_eval_errors_nonfinite_and_duplicate_ids():
    def fail():
        raise RuntimeError('private message')
    results = evaluate([Case('error', fail, float), Case('nan', lambda: 1, lambda _: float('nan'))])
    assert [s.score for s in results] == [0, 0]
    assert results[0].error == 'RuntimeError'
    with pytest.raises(ValueError):
        evaluate([Case('a', lambda: 1, float)] * 2)
    with pytest.raises(ValueError):
        gate([Score('x', float('nan'), False, 0)], results)


@pytest.mark.parametrize('history', [[], [{'state': 'observed', 'tool': 'x'}],
                                     [{'state': 'observed', 'tool': 'delete'}, {'state': 'done'}],
                                     [{'state': 'observed', 'tool': 'save'}, {'state': 'observed', 'tool': 'search'}, {'state': 'done'}]])
def test_trajectory_failure_paths(history):
    assert trajectory_grade(history, required=['search', 'save'], forbidden=['delete']) == 0
