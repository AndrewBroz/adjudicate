import pytest

from adjudicate.llm import ConfigError, merge_layers, resolve_endpoint


def write(xdg, name, body):
    path = xdg / name / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


# --- presets ---------------------------------------------------------------

def test_none_reads_nothing(isolated_llm_config):
    write(isolated_llm_config, "adjudicate", "this is [not toml")
    assert resolve_endpoint("stylefix", "none") is None


def test_unknown_preset_is_an_error():
    with pytest.raises(ConfigError, match="dgx"):
        resolve_endpoint("stylefix", "dgx")


def test_auto_with_nothing_configured_and_no_ollama_is_none():
    assert resolve_endpoint("stylefix") is None


# --- layers ----------------------------------------------------------------

def test_shared_config_is_used(isolated_llm_config):
    write(isolated_llm_config, "adjudicate", 'llm.url = "http://a/v1"\nllm.model = "m1"\n')
    ep = resolve_endpoint("stylefix")
    assert (ep.url, ep.model, ep.name) == ("http://a/v1", "m1", "custom")


def test_app_config_refines_shared_model(isolated_llm_config):
    write(isolated_llm_config, "adjudicate", '[llm]\nurl = "http://a/v1"\nmodel = "m1"\n')
    write(isolated_llm_config, "proofix", '[llm]\nmodel = "m2"\n')
    ep = resolve_endpoint("proofix")
    assert (ep.url, ep.model) == ("http://a/v1", "m2")
    assert resolve_endpoint("stylefix").model == "m1"   # the other app is unaffected


def test_app_env_beats_app_config(isolated_llm_config, monkeypatch):
    write(isolated_llm_config, "stylefix", '[llm]\nurl = "http://a/v1"\nmodel = "a"\n')
    monkeypatch.setenv("STYLEFIX_LLM_MODEL", "b")
    ep = resolve_endpoint("stylefix")
    assert (ep.url, ep.model) == ("http://a/v1", "b")


def test_app_config_beats_shared_env(isolated_llm_config, monkeypatch):
    monkeypatch.setenv("ADJUDICATE_LLM_URL", "http://shared/v1")
    monkeypatch.setenv("ADJUDICATE_LLM_MODEL", "s")
    write(isolated_llm_config, "stylefix", '[llm]\nurl = "http://app/v1"\nmodel = "a"\n')
    assert resolve_endpoint("stylefix").url == "http://app/v1"


def test_flags_beat_everything(isolated_llm_config, monkeypatch):
    monkeypatch.setenv("STYLEFIX_LLM_URL", "http://env/v1")
    monkeypatch.setenv("STYLEFIX_LLM_MODEL", "e")
    ep = resolve_endpoint("stylefix", url="http://flag/v1", model="f", key="k")
    assert (ep.url, ep.model, ep.key) == ("http://flag/v1", "f", "k")


def test_higher_url_discards_lower_server_settings(isolated_llm_config, server):
    write(isolated_llm_config, "adjudicate",
          '[llm]\nurl = "http://dgx/v1"\nmodel = "dgx-model"\napi_key = "k"\n'
          'thinking_switch = true\ntimeout = 30\n')
    ep = resolve_endpoint("stylefix", url=server.url)
    assert ep.model == "served-model"      # discovered, not inherited
    assert ep.key == "" and ep.thinking_switch is False
    assert ep.timeout == 30                # timeout is not server-specific


def test_merge_layers_is_field_by_field():
    merged = merge_layers([{"model": ("hi", "A")},
                           {"url": ("u", "B"), "model": ("lo", "B"), "timeout": (5, "B")}])
    assert merged == {"url": ("u", "B"), "model": ("hi", "A"), "timeout": (5, "B")}


def test_sources_name_each_layer(isolated_llm_config, monkeypatch):
    monkeypatch.setenv("MY_KEY", "k")
    path = write(isolated_llm_config, "adjudicate",
                 '[llm]\nurl = "http://a/v1"\nmodel = "m"\napi_key_env = "MY_KEY"\n')
    monkeypatch.setenv("STYLEFIX_LLM_MODEL", "e")
    ep = resolve_endpoint("stylefix", timeout=9)
    assert ep.sources == {"url": str(path), "model": "STYLEFIX_LLM_MODEL",
                          "api_key": f"{path} (api_key_env MY_KEY)",
                          "timeout": "--timeout", "thinking_switch": "default"}


def test_sources_for_discovery_and_ollama(server, monkeypatch):
    ep = resolve_endpoint("stylefix", url=server.url)
    assert (ep.sources["url"], ep.sources["model"]) == ("--url", "listed by the server")
    assert "api_key" not in ep.sources
    monkeypatch.setenv("OLLAMA_HOST", server.base)
    ep = resolve_endpoint("stylefix")
    assert (ep.sources["url"], ep.sources["model"]) == ("local Ollama", "local Ollama")


# --- keys and validation ---------------------------------------------------

def test_api_key_env(isolated_llm_config, monkeypatch):
    monkeypatch.setenv("MY_KEY", "sekrit")
    write(isolated_llm_config, "adjudicate",
          '[llm]\nurl = "http://a/v1"\nmodel = "m"\napi_key_env = "MY_KEY"\n')
    assert resolve_endpoint("stylefix").key == "sekrit"


def test_api_key_env_unset_is_an_error(isolated_llm_config):
    write(isolated_llm_config, "adjudicate", '[llm]\nurl = "http://a/v1"\napi_key_env = "MY_KEY"\n')
    with pytest.raises(ConfigError, match="MY_KEY"):
        resolve_endpoint("stylefix")


def test_api_key_env_empty_is_an_error(isolated_llm_config, monkeypatch):
    monkeypatch.setenv("MY_KEY", "")
    write(isolated_llm_config, "adjudicate", '[llm]\nurl = "http://a/v1"\napi_key_env = "MY_KEY"\n')
    with pytest.raises(ConfigError, match="MY_KEY"):
        resolve_endpoint("stylefix")


def test_both_key_forms_is_an_error(isolated_llm_config):
    write(isolated_llm_config, "adjudicate", '[llm]\napi_key = "a"\napi_key_env = "B"\n')
    with pytest.raises(ConfigError, match="not both"):
        resolve_endpoint("stylefix")


def test_unknown_key_is_an_error(isolated_llm_config):
    path = write(isolated_llm_config, "stylefix", '[llm]\nurl = "http://a/v1"\nmodle = "m"\n')
    with pytest.raises(ConfigError, match="modle") as e:
        resolve_endpoint("stylefix")
    assert str(path) in str(e.value)


@pytest.mark.parametrize("value", ['"60"', "true"])
def test_wrong_type_is_an_error(isolated_llm_config, value):
    write(isolated_llm_config, "adjudicate", f'[llm]\nurl = "http://a/v1"\ntimeout = {value}\n')
    with pytest.raises(ConfigError, match="timeout"):
        resolve_endpoint("stylefix")


def test_invalid_toml_names_the_file(isolated_llm_config):
    path = write(isolated_llm_config, "adjudicate", "[llm\nurl=")
    with pytest.raises(ConfigError, match="not valid TOML") as e:
        resolve_endpoint("stylefix")
    assert str(path) in str(e.value)


def test_other_tables_are_ignored(isolated_llm_config):
    write(isolated_llm_config, "stylefix", '[llm]\nurl = "http://a/v1"\nmodel = "m"\n[other]\nx = 1\n')
    assert resolve_endpoint("stylefix").model == "m"


# --- timeout ---------------------------------------------------------------

def test_timeout_default_config_and_flag(isolated_llm_config):
    assert resolve_endpoint("stylefix", url="http://a/v1", model="m").timeout == 120.0
    write(isolated_llm_config, "adjudicate", '[llm]\nurl = "http://a/v1"\nmodel = "m"\ntimeout = 30\n')
    assert resolve_endpoint("stylefix").timeout == 30.0
    assert resolve_endpoint("stylefix", timeout=5).timeout == 5.0


# --- model discovery -------------------------------------------------------

def test_model_discovered_from_server(server):
    assert resolve_endpoint("stylefix", url=server.url).model == "served-model"


def test_empty_listing_is_an_error(server):
    server.models = []
    with pytest.raises(ConfigError, match="lists none"):
        resolve_endpoint("stylefix", url=server.url)


def test_listing_down_is_an_error(server):
    server.models_status = 500
    with pytest.raises(ConfigError, match="--model"):
        resolve_endpoint("stylefix", url=server.url)


# --- Ollama ----------------------------------------------------------------

def test_auto_finds_ollama(server, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", server.base)
    ep = resolve_endpoint("stylefix")
    assert (ep.name, ep.url, ep.model) == ("ollama", server.url, "served-model")


def test_auto_with_empty_ollama_is_none(server, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", server.base)
    server.models = []
    assert resolve_endpoint("stylefix") is None


def test_explicit_ollama_with_no_model_is_an_error(server, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", server.base)
    server.models = []
    with pytest.raises(ConfigError):
        resolve_endpoint("stylefix", "ollama")


def test_explicit_ollama_ignores_configured_url(isolated_llm_config, server, monkeypatch):
    write(isolated_llm_config, "adjudicate", '[llm]\nurl = "http://remote/v1"\nmodel = "r"\n')
    monkeypatch.setenv("OLLAMA_HOST", server.base)
    ep = resolve_endpoint("stylefix", "ollama", model="picked")
    assert (ep.name, ep.model) == ("ollama", "picked")
