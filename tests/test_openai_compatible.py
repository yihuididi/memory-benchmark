import json
from types import SimpleNamespace

import pytest

from memory_bench.generators.openai_compatible import OpenAICompatibleGenerator, encode_messages
from memory_bench.types import Message, ModelAdapterRef


def test_multimodal_encoding_preserves_order_and_does_not_mutate_inputs(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"image bytes")
    blocks = ({"type": "text", "text": "before"}, {"type": "image_path", "image_path": str(path)},
              {"type": "text", "text": "after"})
    messages = (Message("system", "system"), Message("user", blocks))
    encoded = encode_messages(messages, (str(path),))
    assert [block["type"] for block in encoded[1]["content"]] == ["text", "image_url", "text", "image_url"]
    assert encoded[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert blocks[1] == {"type": "image_path", "image_path": str(path)}


def test_reader_forwards_settings_usage_and_closes_client(monkeypatch):
    calls, closed = [], []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r"\boxed{A}"), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=9, completion_tokens=2, total_tokens=11))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: closed.append(True))
    monkeypatch.setattr("memory_bench.generators.openai_compatible.create_client", lambda **kwargs: client)
    generator = OpenAICompatibleGenerator(model="vision-reader", base_url="http://localhost:8023/v1")
    settings = {"max_tokens": 20000, "temperature": .6, "top_p": .95, "extra_body": {"top_k": 20}}
    prediction = generator.generate((Message("user", "Question"),), settings=settings)
    assert calls[0] == {"model": "vision-reader", "messages": [{"role": "user", "content": "Question"}], **settings}
    assert prediction.metadata["usage"]["total_tokens"] == 11
    with pytest.raises(ValueError, match="adapters"):
        generator.generate((), settings={}, model_adapter=ModelAdapterRef("lora", "adapter"))
    with pytest.raises(ValueError, match="Unsupported generation settings"):
        generator.generate((), settings={"model": "override"})
    assert len(calls) == 1
    generator.close()
    assert closed == [True]


def test_actual_sdk_request_to_mock_transport(tmp_path, monkeypatch):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    calls = []
    def respond(request):
        calls.append(json.loads(request.content))
        assert str(request.url) == "http://reader.test/v1/chat/completions"
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "vision-reader", "choices": [{"index": 0, "message": {"role": "assistant", "content": r"\boxed{A}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
    client = openai.OpenAI(base_url="http://reader.test/v1", api_key="test",
                           http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr("memory_bench.generators.openai_compatible.create_client", lambda **kwargs: client)
    image = tmp_path / "question.png"
    image.write_bytes(b"image")
    generator = OpenAICompatibleGenerator(model="vision-reader", base_url="http://reader.test/v1")
    try:
        output = generator.generate((Message("user", "q"),), attachments=(str(image),),
                                    settings={"max_tokens": 20000, "extra_body": {"top_k": 20}})
        assert output.answer == r"\boxed{A}"
        assert output.metadata["usage"]["total_tokens"] == 12
        assert calls[0]["top_k"] == 20  # SDK merges extra_body at the wire boundary.
        assert calls[0]["messages"][0]["content"][1]["type"] == "image_url"
    finally:
        generator.close()
    assert client.is_closed()
