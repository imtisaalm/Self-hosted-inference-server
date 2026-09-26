import asyncio
import json
import sys

import pytest

from aieng.mcp import Client, MCPTool, PREFIX, Server, VERSION, default_server, validator
from aieng.workflow import Busy, LostLease, Step, WorkflowEngine


def test_workflow_resumes_after_process_restart(tmp_path):
    path = tmp_path / 'runs.db'
    calls, keys = [], []
    fail = [True]
    def first(state, key):
        calls.append('first')
        return state + 1
    def second(state, key):
        keys.append(key)
        if fail[0]:
            raise ConnectionError('retry')
        return state * 2
    steps = [Step('add', '1', first), Step('double', '1', second)]
    with WorkflowEngine(path) as engine:
        with pytest.raises(ConnectionError):
            engine.run('job', 2, steps)
        assert engine.inspect('job')['next_step'] == 1
    fail[0] = False
    with WorkflowEngine(path) as engine:
        assert engine.run('job', 2, steps) == 6
        assert engine.run('job', 2, steps) == 6
        assert engine.inspect('job')['status'] == 'complete'
        with pytest.raises(ValueError):
            engine.run('job', 3, steps)
        with pytest.raises(ValueError):
            engine.run('job', 2, [Step('add', '2', first)])
    assert calls == ['first'] and keys[0] == keys[1]


def test_workflow_live_lease_blocks_second_worker(tmp_path):
    path = tmp_path / 'lease.db'
    with WorkflowEngine(path) as a, WorkflowEngine(path) as b:
        steps = []
        def first(state, key):
            with pytest.raises(Busy):
                b.run('job', 0, steps)
            return 1
        steps.append(Step('work', '1', first))
        assert a.run('job', 0, steps) == 1


def test_workflow_stale_worker_cannot_commit(tmp_path):
    path, now = tmp_path / 'lease.db', [0.0]
    with WorkflowEngine(path, clock=lambda: now[0]) as stale, WorkflowEngine(path, clock=lambda: now[0]) as fresh:
        def old_step(state, key):
            now[0] = 2
            assert fresh.run('job', 0, [Step('work', '1', lambda s, k: 99)], lease_s=1) == 99
            return 1
        with pytest.raises(LostLease):
            stale.run('job', 0, [Step('work', '1', old_step)], lease_s=1)
        assert stale.inspect('job')['state'] == 99
        assert stale.inspect('job')['status'] == 'complete'


def request(method, **params):
    params['_meta'] = {PREFIX + 'protocolVersion': VERSION, PREFIX + 'clientCapabilities': {}}
    return {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}


@pytest.mark.parametrize('line,code', [('{broken', -32700), ('[]', -32600), ('{"jsonrpc":"1.0"}', -32600),
                                      ('{"jsonrpc":"2.0","id":true,"method":"ping"}', -32600),
                                      ('{"jsonrpc":"2.0","id":1,"method":"ping"}', -32602)])
def test_rpc_errors(line, code):
    assert default_server().handle(line)['error']['code'] == code


def test_rpc_discovery_versions_notifications_errors():
    server = default_server()
    result = server.handle(json.dumps(request('server/discover')))['result']
    assert result['supportedVersions'] == [VERSION] and result['capabilities'] == {'tools': {}}
    bad = request('ping')
    bad['params']['_meta'][PREFIX + 'protocolVersion'] = '1900-01-01'
    error = server.handle(json.dumps(bad))['error']
    assert error['code'] == -32022 and error['data']['supported'] == [VERSION]
    assert server.handle('{"jsonrpc":"2.0","method":"notifications/test"}') is None
    assert server.handle(json.dumps(request('unknown')))['error']['code'] == -32601


def test_rpc_tool_validation_output_and_rate_limit():
    s = default_server()
    ok = s.handle(json.dumps(request('tools/call', name='arithmetic', arguments={'expression': '6*7'})))
    assert ok['result']['structuredContent']['value'] == 42
    wrong = s.handle(json.dumps(request('tools/call', name='arithmetic', arguments={'expression': 4})))
    assert wrong['result']['isError']
    unknown = s.handle(json.dumps(request('tools/call', name='missing')))
    assert unknown['error']['code'] == -32602
    for _ in range(100):
        last = s.handle(json.dumps(request('tools/call', name='arithmetic', arguments={'expression': '1'})))
    assert last['result']['isError'] and 'rate limit' in last['result']['content'][0]['text']


def test_rpc_invalid_output_is_tool_error_and_remote_refs_fail_closed():
    s = Server([MCPTool('bad', 'bad output', {'type': 'object'}, lambda _: 'wrong', {'type': 'number'})])
    assert s.handle(json.dumps(request('tools/call', name='bad')))['result']['isError']
    with pytest.raises(Exception):
        validator({'$ref': 'https://invalid.example/schema.json'}).validate({})


def test_stdio_actual_subprocess_client_server():
    async def run():
        async with Client([sys.executable, '-m', 'aieng.mcp']) as client:
            tools = await client.request('tools/list')
            assert tools['tools'][0]['name'] == 'arithmetic'
            result = await client.request('tools/call', name='arithmetic', arguments={'expression': '(8+6)*3'})
            assert result['structuredContent'] == {'value': 42}
            result = await client.request('tools/call', name='arithmetic', arguments={'expression': 'open("secret")'})
            assert result['isError']
        assert client.process is None
    asyncio.run(run())


def test_stdio_timeout_terminates_process():
    async def run():
        client = Client([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=.02)
        with pytest.raises(TimeoutError):
            async with client:
                pass
        assert client.process is None
    asyncio.run(run())
