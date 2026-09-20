from typing import Any

import pytest

from inference_server.client import _content_from_chunk


def test_extracts_chat_delta_content() -> None:
    payload: dict[str, Any] = {"choices": [{"delta": {"content": "token"}}]}
    assert _content_from_chunk(payload) == "token"


def test_empty_delta_is_ignored() -> None:
    payload: dict[str, Any] = {"choices": [{"delta": {}}]}
    assert _content_from_chunk(payload) == ""


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": "invalid"},
        {"choices": [None]},
        {"choices": [{"delta": "invalid"}]},
        {"choices": [{"delta": {"content": 123}}]},
    ],
)
def test_invalid_chunk_shapes_are_ignored(payload: dict[str, Any]) -> None:
    assert _content_from_chunk(payload) == ""
