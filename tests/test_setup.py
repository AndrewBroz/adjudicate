import io
import os
import stat
import tomllib

import pytest

from adjudicate.setup import check_endpoint, run_setup


def setup(answers, app="stylefix"):
    """Run setup on scripted answers, one per prompt; returns (status, output)."""
    out = io.StringIO()
    status = run_setup(app, io.StringIO("".join(a + "\n" for a in answers)), out,
                       require_tty=False)
    return status, out.getvalue()


def shared(xdg):
    return xdg / "adjudicate" / "config.toml"


def llm(path):
    return tomllib.loads(path.read_text())["llm"]


# --- the three paths --------------------------------------------------------

def test_self_hosted_detects_thinking_switch(isolated_llm_config, server):
    # kind 3 (self-hosted), url, key 3 (none), model 1, save 1 (shared)
    status, out = setup(["3", server.url, "3", "1", "1"])
    assert status == 0, out
    assert llm(shared(isolated_llm_config)) == {
        "url": server.url, "model": "served-model", "thinking_switch": True}
    assert server.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking_switch = true" in out


def test_self_hosted_that_rejects_thinking_switch(isolated_llm_config, server):
    server.reject = {"chat_template_kwargs"}
    status, out = setup(["3", server.url, "3", "1", "1"])
    assert status == 0, out
    assert llm(shared(isolated_llm_config))["thinking_switch"] is False


def test_hosted_with_env_key(isolated_llm_config, server, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    # kind 2 (hosted), url, key 1 (env var), var name, model 1, save 1
    status, out = setup(["2", server.url, "1", "TEST_KEY", "1", "1"])
    assert status == 0, out
    assert llm(shared(isolated_llm_config)) == {
        "url": server.url, "model": "served-model", "api_key_env": "TEST_KEY"}
    assert server.auth[0] == "Bearer k"              # the listing was asked with the key
    assert "chat_template_kwargs" not in server.requests[0]


def test_ollama_saved_to_app_file_as_explicit_url(isolated_llm_config, server, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", server.base)
    # kind 1 (Ollama), model 1, save 2 (stylefix only)
    status, out = setup(["1", "1", "2"])
    assert status == 0, out
    assert llm(isolated_llm_config / "stylefix" / "config.toml") == {
        "url": server.url, "model": "served-model"}
    assert not shared(isolated_llm_config).exists()


def test_ollama_not_running(isolated_llm_config):
    status, out = setup(["1"])
    assert status == 1 and "ollama serve" in out
    assert not shared(isolated_llm_config).exists()


def test_typed_model_name(isolated_llm_config, server):
    # model 2 = "Other (type a name)"
    status, out = setup(["3", server.url, "3", "2", "my-model", "1"])
    assert status == 0, out
    assert llm(shared(isolated_llm_config))["model"] == "my-model"


# --- URL handling -------------------------------------------------------------

def test_url_without_scheme_is_asked_again(isolated_llm_config, server):
    status, out = setup(["3", "gpu:8000/v1", server.base + "/v1/chat/completions/", "3", "1", "1"])
    assert status == 0, out
    assert "include the scheme" in out
    assert llm(shared(isolated_llm_config))["url"] == server.url


def test_v1_offered_and_appended(isolated_llm_config, server):
    status, out = setup(["3", server.base, "y", "3", "1", "1"])
    assert status == 0, out
    assert llm(shared(isolated_llm_config))["url"] == server.url


def test_unlistable_server_asks_to_continue(isolated_llm_config, server):
    server.models_status = 500
    status, out = setup(["3", server.url, "3", "n"])
    assert status == 1 and "could not list models" in out
    assert not shared(isolated_llm_config).exists()


def test_unlistable_server_says_why(isolated_llm_config, server):
    server.models_status = 401
    status, out = setup(["2", server.url, "3", "n"])
    assert status == 1
    assert f"could not list models at {server.url}/models: HTTP 401" in out
    assert "down" in out                       # the start of the body


def test_unreachable_server_says_why(isolated_llm_config):
    status, out = setup(["3", "http://127.0.0.1:9/v1", "3", "n"])
    assert status == 1
    line = next(l for l in out.splitlines() if "could not list models" in l)
    assert line.split("/models: ", 1)[1].strip()   # a reason follows


def test_server_rejecting_only_response_format_keeps_thinking_switch(isolated_llm_config, server):
    server.reject = {"response_format"}
    status, out = setup(["3", server.url, "3", "1", "1"])
    assert status == 0, out
    assert llm(shared(isolated_llm_config))["thinking_switch"] is True
    assert "response_format" not in server.requests[0]


# --- keys -----------------------------------------------------------------------

def test_pasted_key_is_saved_0600_and_never_shown(isolated_llm_config, server):
    status, out = setup(["2", server.url, "2", "sekrit", "1", "1"])
    assert status == 0, out
    path = shared(isolated_llm_config)
    assert llm(path)["api_key"] == "sekrit"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert "sekrit" not in out


def test_pasted_key_over_an_existing_0644_file(isolated_llm_config, server):
    path = shared(isolated_llm_config)
    path.parent.mkdir(parents=True)
    path.write_text(OLD)
    os.chmod(path, 0o644)
    status, out = setup(["2", server.url, "2", "sekrit", "1", "1", "y"])
    assert status == 0, out
    assert llm(path)["api_key"] == "sekrit"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    left = sorted(p.name for p in path.parent.iterdir())
    assert left == ["config.toml", "config.toml.bak"], left


def test_saved_file_is_never_briefly_world_readable(isolated_llm_config, server, monkeypatch):
    # The file is created with 0600 from the start: check the mode at the
    # moment it takes the config's name.
    import adjudicate.setup as setup_module
    modes = []
    real_replace = os.replace

    def spy(src, dst):
        modes.append(stat.S_IMODE(os.stat(src).st_mode))
        return real_replace(src, dst)
    monkeypatch.setattr(setup_module.os, "replace", spy)
    status, out = setup(["2", server.url, "2", "sekrit", "1", "1"])
    assert status == 0, out
    assert modes == [0o600]


def test_without_a_key_an_existing_mode_is_kept(isolated_llm_config, server):
    path = shared(isolated_llm_config)
    path.parent.mkdir(parents=True)
    path.write_text(OLD)
    os.chmod(path, 0o640)
    status, out = setup(["3", server.url, "3", "1", "1", "y"])
    assert status == 0, out
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640


def test_failed_write_leaves_no_temporary_file(isolated_llm_config, server, monkeypatch):
    import adjudicate.setup as setup_module

    def broken(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(setup_module.os, "replace", broken)
    status, out = setup(["3", server.url, "3", "1", "1"])
    assert status == 1 and "cannot write" in out and "disk full" in out
    path = shared(isolated_llm_config)
    assert list(path.parent.iterdir()) == []


def test_unreadable_existing_file_is_not_edited(isolated_llm_config, server):
    path = shared(isolated_llm_config)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"[llm]\nmodel = \"\xff\"\n")
    status, out = setup(["3", server.url, "3", "1", "1"])
    assert status == 1
    assert f"{path}: cannot read" in out and "by hand" in out
    assert path.read_bytes() == b"[llm]\nmodel = \"\xff\"\n"
    assert not path.with_name("config.toml.bak").exists()


def test_unset_env_key_warns(isolated_llm_config, server):
    status, out = setup(["2", server.url, "1", "NOT_SET_ANYWHERE", "1", "1"])
    assert "NOT_SET_ANYWHERE is not set" in out
    # saved even so; the check afterwards fails with the ConfigError for the unset variable
    assert llm(shared(isolated_llm_config))["api_key_env"] == "NOT_SET_ANYWHERE"
    assert status == 2


# --- existing files -------------------------------------------------------------

OLD = '[llm]\nurl = "http://old/v1"\nmodel = "old"\n\n[other]\nx = 1\n'


def test_replace_keeps_other_tables_and_backs_up(isolated_llm_config, server):
    path = shared(isolated_llm_config)
    path.parent.mkdir(parents=True)
    path.write_text(OLD)
    status, out = setup(["3", server.url, "3", "1", "1", "y"])
    assert status == 0, out
    assert "already has" in out and "http://old/v1" in out
    assert path.with_name("config.toml.bak").read_text() == OLD
    data = tomllib.loads(path.read_text())
    assert data["llm"]["url"] == server.url and data["other"] == {"x": 1}


def test_declining_replace_changes_nothing(isolated_llm_config, server):
    path = shared(isolated_llm_config)
    path.parent.mkdir(parents=True)
    path.write_text(OLD)
    status, out = setup(["3", server.url, "3", "1", "1", "n"])
    assert status == 1
    assert path.read_text() == OLD
    assert not path.with_name("config.toml.bak").exists()


def test_other_tables_only_gets_llm_appended(isolated_llm_config, server):
    path = shared(isolated_llm_config)
    path.parent.mkdir(parents=True)
    path.write_text("[other]\nx = 1\n")
    status, out = setup(["3", server.url, "3", "1", "1"])
    assert status == 0, out
    data = tomllib.loads(path.read_text())
    assert data["other"] == {"x": 1} and data["llm"]["url"] == server.url


def test_dotted_keys_are_not_edited(isolated_llm_config, server):
    path = shared(isolated_llm_config)
    path.parent.mkdir(parents=True)
    path.write_text('llm.url = "http://old/v1"\n')
    status, out = setup(["3", server.url, "3", "1", "1"])
    assert status == 1 and "by hand" in out and f'url = "{server.url}"' in out
    assert path.read_text() == 'llm.url = "http://old/v1"\n'


# --- cancelling and terminals -----------------------------------------------------

def test_end_of_input_cancels(isolated_llm_config):
    status, out = setup(["3"])            # input ends at the URL prompt
    assert status == 1 and "cancelled" in out
    assert not shared(isolated_llm_config).exists()


def test_refuses_without_a_terminal(isolated_llm_config):
    out, err = io.StringIO(), io.StringIO()
    assert run_setup("stylefix", io.StringIO(""), out, err=err) == 2
    assert "terminal" in err.getvalue() and out.getvalue() == ""


def test_refusal_goes_to_stderr_by_default(isolated_llm_config, capsys):
    assert run_setup("stylefix", io.StringIO("")) == 2
    captured = capsys.readouterr()
    assert "terminal" in captured.err and "terminal" not in captured.out


def test_interrupt_during_the_closing_check_says_the_file_was_saved(
        isolated_llm_config, server, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr("adjudicate.setup.check_endpoint", interrupted)
    status, out = setup(["3", server.url, "3", "1", "1"])
    path = shared(isolated_llm_config)
    assert status == 1
    assert f"Saved {path}; check interrupted." in out
    assert "nothing was written" not in out
    assert llm(path)["url"] == server.url


# --- check_endpoint ---------------------------------------------------------------

def check(**kw):
    """Run the check; returns (status, stdout and stderr together)."""
    out, err = io.StringIO(), io.StringIO()
    status = check_endpoint("stylefix", out, err=err, **kw)
    return status, out.getvalue() + err.getvalue()


def test_check_with_nothing_configured():
    status, out = check()
    assert status == 1 and "stylefix --setup" in out


def test_check_config_error(isolated_llm_config):
    path = shared(isolated_llm_config)
    path.parent.mkdir(parents=True)
    path.write_text("[llm\n")
    out, err = io.StringIO(), io.StringIO()
    assert check_endpoint("stylefix", out, err=err) == 2
    assert str(path) in err.getvalue() and out.getvalue() == ""


def test_check_config_error_goes_to_stderr_by_default(isolated_llm_config, capsys):
    path = shared(isolated_llm_config)
    path.parent.mkdir(parents=True)
    path.write_text("[llm\n")
    assert check_endpoint("stylefix") == 2
    captured = capsys.readouterr()
    assert str(path) in captured.err and str(path) not in captured.out


def test_check_prints_sources_and_hides_the_key(isolated_llm_config, server, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sekrit")
    path = shared(isolated_llm_config)
    path.parent.mkdir(parents=True)
    path.write_text(f'[llm]\nurl = "{server.url}"\nmodel = "m"\napi_key_env = "TEST_KEY"\n')
    status, out = check()
    assert status == 0, out
    lines = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert server.url in lines["url"] and str(path) in lines["url"]
    assert "(hidden)" in lines["api_key"] and "api_key_env TEST_KEY" in lines["api_key"]
    assert "default" in lines["timeout"]
    assert "sekrit" not in out
    assert "chat ✓" in out


def test_check_endpoint_that_does_not_answer():
    status, out = check(url="http://127.0.0.1:9/v1", model="m")
    assert status == 1 and "chat ✗" in out


def test_write_works_where_os_has_no_fchmod(tmp_path, monkeypatch):
    # Windows before Python 3.13 has no os.fchmod; the save must still work there.
    from adjudicate.setup import _write_private
    monkeypatch.delattr("os.fchmod", raising=False)
    path = tmp_path / "config.toml"
    _write_private(path, '[llm]\nurl = "http://a/v1"\n', 0o600)
    assert path.read_text() == '[llm]\nurl = "http://a/v1"\n'
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]
