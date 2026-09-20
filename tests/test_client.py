from inference_server.client import _content_from_chunk


def test_extracts_chat_delta_content() -> None:
    payload = {"choices": [{"delta": {"content": "token"}}]}
    assert _content_from_chunk(payload) == "token"


def test_empty_delta_is_ignored() -> None:
    assert _content_from_chunk({"choices": [{"delta": {}}]}) == ""
