import os

import pytest

from adjudicate.llm import Endpoint, LLMError


def test_isolation_hides_real_config(isolated_llm_config):
    assert os.environ["XDG_CONFIG_HOME"] == str(isolated_llm_config)
    assert not any(os.environ.get(f"{app}_LLM_{s}") for app in ("ADJUDICATE", "STYLEFIX", "PROOFIX")
                   for s in ("URL", "MODEL", "KEY"))


def test_default_body_has_no_vllm_extension(server):
    Endpoint(server.url, "m").chat("s", "u")
    body = server.requests[-1]
    assert "chat_template_kwargs" not in body
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == "m" and body["temperature"] == 0


def test_thinking_switch_sends_enable_thinking_false(server):
    Endpoint(server.url, "m", thinking_switch=True).chat("s", "u")
    assert server.requests[-1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_think_sends_enable_thinking_true(server):
    Endpoint(server.url, "m", think=True).chat("s", "u")
    assert server.requests[-1]["chat_template_kwargs"] == {"enable_thinking": True}


def test_rejected_extension_is_dropped_once_for_the_run(server):
    server.reject = {"chat_template_kwargs"}
    ep = Endpoint(server.url, "m", thinking_switch=True)
    ep.chat("s", "u")
    ep.chat("s", "u")
    # first call: rejected, then retried without; second call: sent without
    assert len(server.requests) == 3
    assert ["chat_template_kwargs" in b for b in server.requests] == [True, False, False]
    assert all("response_format" in b for b in server.requests)


def test_both_optional_fields_dropped_in_order(server):
    server.reject = {"chat_template_kwargs", "response_format"}
    ep = Endpoint(server.url, "m", thinking_switch=True)
    assert ep.chat("s", "u") == '{"decisions": []}'
    assert [sorted(b.keys() & server.reject) for b in server.requests] == [
        ["chat_template_kwargs", "response_format"], ["response_format"], []]


def test_400_with_nothing_to_drop_raises(server):
    server.reject = {"messages"}
    with pytest.raises(LLMError, match="HTTP 400"):
        Endpoint(server.url, "m").chat("s", "u")
    assert len(server.requests) == 2   # with response_format, then without it
