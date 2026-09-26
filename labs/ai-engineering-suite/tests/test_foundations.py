import asyncio
import math
from dataclasses import replace

import pytest

from aieng.cache import CacheScope, SemanticCache
from aieng.context import ContextAssembler, ContextItem
from aieng.core import canonical, number, positive_int
from aieng.retrieval import BM25, DenseIndex, RetrievalStack, TfidfEncoder, chunk_document, cosine
from aieng.router import Model, ModelRouter, RoutingFailed


@pytest.mark.parametrize('value', [True, -1, float('nan'), float('inf'), '2', None])
def test_numeric_validation(value):
    with pytest.raises(ValueError):
        number(value, 'x')


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_positive_int(value):
    with pytest.raises(ValueError):
        positive_int(value, 'x')


def test_canonical_json():
    assert canonical({'b': 1, 'a': 2}) == canonical({'a': 2, 'b': 1})
    with pytest.raises(ValueError):
        canonical(float('nan'))


def test_context_dependency_budget():
    assembler = ContextAssembler(count=len, framing=0)
    items = [ContextItem('system', 'safe', pinned=True),
             ContextItem('call', 'call', priority=1),
             ContextItem('result', '42', priority=10, requires=('call',)),
             ContextItem('old', 'old old old', priority=0)]
    result = assembler.assemble(items, limit=12, reserve=2)
    assert [x.id for x in result.items] == ['system', 'call', 'result']
    assert result.used == 10 and result.available == 10
    assert result.omitted == ('old',)


@pytest.mark.parametrize('items', [
    [ContextItem('a', 'x'), ContextItem('a', 'y')],
    [ContextItem('a', 'x', requires=('missing',))],
    [ContextItem('a', 'x', requires=('b',)), ContextItem('b', 'y', requires=('a',))],
])
def test_context_rejects_invalid_graph(items):
    with pytest.raises(ValueError):
        ContextAssembler().assemble(items, limit=100)


def test_context_pinned_never_silently_dropped():
    with pytest.raises(ValueError):
        ContextAssembler().assemble([ContextItem('s', 'too long', pinned=True)], limit=2)


def test_context_unicode_byte_budget():
    out = ContextAssembler(framing=0).assemble([ContextItem('a', '你好')], limit=6)
    assert out.used == 6


def test_chunk_offsets_overlap_no_tail_duplicates():
    text = 'alpha  beta\ngamma delta epsilon'
    chunks = chunk_document('doc', text, size=3, overlap=1)
    assert len(chunks) == 2
    assert all(text[c.start:c.end] == c.text for c in chunks)
    assert chunks[1].text.startswith('gamma')


@pytest.mark.parametrize('size,overlap', [(0, 0), (2, 2), (2, -1), (True, 0)])
def test_chunk_bad_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_document('d', 'a b c', size=size, overlap=overlap)


def test_bm25_hybrid_dense():
    docs = ['a robot uses sensors', 'a cache stores keys values', 'a router chooses a model']
    chunks = [chunk_document(str(i), s)[0] for i, s in enumerate(docs)]
    bm25 = BM25(chunks)
    assert bm25.search('cache')[0][0] == 1
    assert bm25.search('unseen') == []
    encoder = TfidfEncoder(docs)
    chunks = [chunk_document(str(i), s)[0] for i, s in enumerate(docs)]
    stack = RetrievalStack(chunks, encoder)
    assert stack.search('cache keys')[0].chunk.source == '1'
    assert stack.search('') == []
    assert DenseIndex([[1, 0], [0, 1]]).search([0, 1])[0][0] == 1


@pytest.mark.parametrize('left,right', [([0, 0], [1, 0]), ([math.nan], [1]),
                                       ([1, 2], [1]), ([], [])])
def test_vectors_invalid(left, right):
    with pytest.raises(ValueError):
        cosine(left, right)


def test_cache_scope_ttl_and_metrics(tmp_path):
    now = [10.0]
    scope = CacheScope('tenant-a', 'model', '1', 'e1', 'docs1')
    path = tmp_path / 'cache.db'
    with SemanticCache(path, clock=lambda: now[0]) as cache:
        cache.put(scope, 'question', [1, 0], {'answer': 42}, ttl=10)
        assert cache.get(scope, 'paraphrase', [0.99, 0.01]).value == {'answer': 42}
        assert cache.get(replace(scope, tenant='tenant-b'), 'question', [1, 0]) is None
        assert cache.get(replace(scope, prompt_version='2'), 'question', [1, 0]) is None
        assert cache.get(scope, 'different', [0, 1]) is None
        assert cache.stats(scope)['hit_rate'] == 0.5
    with SemanticCache(path, clock=lambda: now[0]) as cache:
        assert cache.get(scope, 'question', [1, 0]).exact
        now[0] = 20
        assert cache.get(scope, 'question', [1, 0]) is None


def test_cache_lru_and_dimension_guard():
    now = [0.0]
    s = CacheScope('t', 'm', 'p', 'e', 'c')
    with SemanticCache(capacity=2, clock=lambda: now[0]) as c:
        c.put(s, 'a', [1, 0], 'a')
        now[0] += 1
        c.put(s, 'b', [0, 1], 'b')
        now[0] += 1
        c.get(s, 'a', [1, 0])
        now[0] += 1
        c.put(s, 'c', [-1, 0], 'c')
        assert c.get(s, 'b', [0, 1], threshold=1) is None
        with pytest.raises(ValueError):
            c.put(s, 'd', [1, 2, 3], 'd')
        assert c.invalidate(s) == 2


def test_router_fallback_and_aggregate_budget():
    models = [Model('cheap', 1, 1, 10, 0.8), Model('better', 2, 2, 20, 0.95)]
    seen = []
    async def provider(model, cap):
        seen.append((model.name, cap))
        if model.name == 'cheap':
            raise ConnectionError('failed')
        return 'answer'
    router = ModelRouter(models)
    result = asyncio.run(router.run(provider, input_tokens=100, output_tokens=100, budget=0.0006))
    assert result.model == 'better' and result.attempts == ('cheap', 'better')
    assert result.reserved_cost == pytest.approx(0.0006)
    with pytest.raises(RoutingFailed) as error:
        asyncio.run(ModelRouter(models).run(provider, input_tokens=100, output_tokens=100, budget=0.0005))
    assert error.value.attempts == ('cheap',)


def test_router_circuit_and_quality():
    models = [Model('a', 1, 1, 10, 0.5), Model('b', 2, 2, 20, 0.9)]
    now = [0.0]
    router = ModelRouter(models, failures=1, cooldown=3, clock=lambda: now[0])
    calls = []
    async def provider(m, cap):
        calls.append(m.name)
        if m.name == 'a':
            raise OSError()
        return 'ok'
    for _ in range(2):
        asyncio.run(router.run(provider, input_tokens=10, output_tokens=10, budget=1))
    assert calls == ['a', 'b', 'b']
    now[0] = 4
    asyncio.run(router.run(provider, input_tokens=10, output_tokens=10, budget=1))
    assert calls[-2:] == ['a', 'b']
    asyncio.run(router.run(provider, input_tokens=10, output_tokens=10, budget=1, min_quality=.8))
    assert calls[-1] == 'b'


def test_router_deadline_and_programming_errors():
    r = ModelRouter([Model('a', 1, 1, 10, 1)])
    async def slow(m, cap):
        await asyncio.sleep(10)
    with pytest.raises(RoutingFailed):
        asyncio.run(r.run(slow, input_tokens=1, output_tokens=1, budget=1, deadline_s=.005))
    async def bug(m, cap):
        raise ValueError('programming error')
    with pytest.raises(ValueError):
        asyncio.run(r.run(bug, input_tokens=1, output_tokens=1, budget=1))
