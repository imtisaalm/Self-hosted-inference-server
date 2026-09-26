"""Opt-in integration checks. CI provisions the declared image before enabling them."""
import os

import pytest

from aieng.sandbox import DockerExecutor, ExecutionLimit

pytestmark = pytest.mark.skipif(os.environ.get('AIENG_DOCKER_TESTS') != '1', reason='Docker runtime explicitly required')


def test_docker_execution_and_read_only_root():
    executor = DockerExecutor(timeout_s=10)
    success = executor.execute('print(6 * 7)')
    assert success.returncode == 0 and success.stdout.strip() == '42'
    blocked = executor.execute('open("/cannot-write", "w").write("x")')
    assert blocked.returncode != 0


def test_docker_network_is_disabled():
    result = DockerExecutor(timeout_s=10).execute(
        'import socket\ns=socket.socket()\ns.settimeout(1)\ns.connect(("1.1.1.1", 443))')
    assert result.returncode != 0


def test_docker_infinite_loop_is_terminated():
    with pytest.raises(ExecutionLimit):
        DockerExecutor(timeout_s=2).execute('while True: pass')


def test_docker_output_is_bounded():
    with pytest.raises(ExecutionLimit):
        DockerExecutor(timeout_s=10, output_bytes=1000).execute('print("x" * 10000)')
