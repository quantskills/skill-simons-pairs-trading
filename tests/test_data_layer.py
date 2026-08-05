from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import data_layer  # noqa: E402
import pandadata_security as security  # noqa: E402


@pytest.fixture(autouse=True)
def _provider_tls_is_healthy(monkeypatch):
    monkeypatch.setattr(
        security,
        "_probe_tls",
        lambda base_url, timeout=5.0: {
            "tls_version": "TLSv1.3",
            "certificate_not_after": "Dec 31 23:59:59 2030 GMT",
        },
    )


def test_default_credential_file_is_user_level_only():
    assert data_layer._env_candidates() == [Path.home() / ".pandadata.env"]


def test_import_does_not_create_cache_directory(tmp_path):
    cache_dir = tmp_path / "not-created-on-import"
    env = os.environ.copy()
    env["SIMONS_CACHE"] = str(cache_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0, {str(ROOT / 'scripts')!r});"
                "import data_layer"
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert not cache_dir.exists()


def test_dotenv_only_loads_credential_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PANDADATA_USER=test_user\n"
        "PANDADATA_PASSWORD=secret\n"
        "UNRELATED_DANGEROUS_KEY=must_not_load\n",
        encoding="utf-8",
    )
    for key in ("PANDADATA_USER", "PANDADATA_PASSWORD", "UNRELATED_DANGEROUS_KEY"):
        monkeypatch.delenv(key, raising=False)

    data_layer._load_dotenv([env_file])

    assert os.environ["PANDADATA_USER"] == "test_user"
    assert os.environ["PANDADATA_PASSWORD"] == "secret"
    assert "UNRELATED_DANGEROUS_KEY" not in os.environ


def test_direct_login_check_never_prints_token(monkeypatch, capsys):
    token = "sensitive-token-value"
    monkeypatch.setattr(data_layer, "ensure_login", lambda: token)

    data_layer.print_login_status()

    output = capsys.readouterr().out
    assert token[:8] not in output
    assert "OK" in output


def test_login_retries_transient_failure_without_printing_secret(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(data_layer, "_TOKEN", None)
    monkeypatch.setattr(data_layer, "_load_dotenv", lambda candidates=None: None)
    monkeypatch.setattr(data_layer.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("PANDADATA_USER", "test_user")
    monkeypatch.setenv("PANDADATA_PASSWORD", "do-not-print-this")

    def fake_session(module, *, username, password, token):
        calls.append((username, password, token))
        if len(calls) < 3:
            raise RuntimeError("temporary timeout")
        return "token-value"

    monkeypatch.setattr(data_layer, "establish_session", fake_session)

    assert data_layer.ensure_login() == "token-value"
    assert len(calls) == 3
    assert "do-not-print-this" not in capsys.readouterr().out


def test_direct_login_failure_is_sanitized(monkeypatch, capsys):
    secret = "do-not-print-this"
    monkeypatch.setattr(
        data_layer, "ensure_login",
        lambda: (_ for _ in ()).throw(RuntimeError(f"failure {secret}")),
    )

    rc = data_layer.print_login_status()

    output = capsys.readouterr().out
    assert rc != 0
    assert secret not in output
    assert "FAIL" in output


def test_login_retries_empty_token_then_succeeds(monkeypatch):
    calls = []
    monkeypatch.setattr(data_layer, "_TOKEN", None)
    monkeypatch.setattr(data_layer, "_load_dotenv", lambda candidates=None: None)
    monkeypatch.setattr(data_layer.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("PANDADATA_USER", "test_user")
    monkeypatch.setenv("PANDADATA_PASSWORD", "do-not-print-this")

    def fake_session(module, *, username, password, token):
        calls.append((username, password, token))
        if len(calls) < 3:
            raise RuntimeError("empty token")
        return "token-value"

    monkeypatch.setattr(data_layer, "establish_session", fake_session)

    assert data_layer.ensure_login() == "token-value"
    assert len(calls) == 3


def test_industry_map_uses_bounded_concurrency_and_preserves_order(monkeypatch):
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def fake_get_stock_industry(stock_symbol: str, level: str = "L1"):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.03)
        with lock:
            state["active"] -= 1
        return pd.DataFrame({"industry_name": [f"industry-{stock_symbol}"]})

    monkeypatch.setattr(data_layer, "get_stock_industry", fake_get_stock_industry)
    symbols = [f"S{i}" for i in range(12)]

    result = data_layer.get_industry_map(symbols, level="L1", max_workers=4)

    assert 1 < state["peak"] <= 4
    assert result["symbol"].tolist() == symbols
    assert result["industry"].tolist() == [f"industry-{s}" for s in symbols]


def test_industry_map_retries_each_transient_failure_before_marking_unknown(
        monkeypatch):
    calls = {}
    monkeypatch.setattr(data_layer.time, "sleep", lambda seconds: None)

    def fake_get_stock_industry(stock_symbol: str, level: str = "L1"):
        calls[stock_symbol] = calls.get(stock_symbol, 0) + 1
        if calls[stock_symbol] < 3:
            raise RuntimeError("temporary service error")
        return pd.DataFrame({
            "industry_name": [f"industry-{stock_symbol}"]
        })

    monkeypatch.setattr(
        data_layer, "get_stock_industry", fake_get_stock_industry
    )

    result = data_layer.get_industry_map(
        ["A", "B"], level="L1", max_workers=1
    )

    assert calls == {"A": 3, "B": 3}
    assert result.to_dict("records") == [
        {"symbol": "A", "industry": "industry-A"},
        {"symbol": "B", "industry": "industry-B"},
    ]


def test_industry_map_stops_after_three_persistent_failures(monkeypatch):
    calls = []
    monkeypatch.setattr(data_layer.time, "sleep", lambda seconds: None)

    def fake_get_stock_industry(stock_symbol: str, level: str = "L1"):
        calls.append(stock_symbol)
        raise RuntimeError("persistent service error")

    monkeypatch.setattr(
        data_layer, "get_stock_industry", fake_get_stock_industry
    )

    result = data_layer.get_industry_map(
        ["A"], level="L1", max_workers=1
    )

    assert calls == ["A", "A", "A"]
    assert result.to_dict("records") == [
        {"symbol": "A", "industry": "UNKNOWN"}
    ]


def _adjusted_daily_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": ["20260105", "20260106"],
        "symbol": ["A", "A"],
        "close": [10.0, 10.5],
    })


def test_adjusted_daily_defaults_to_panda_data_pre_adjusted_api(
        monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(data_layer, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_layer, "ensure_login", lambda: "ok")
    monkeypatch.setattr(
        data_layer.pdd,
        "get_stock_daily_pre",
        lambda **kwargs: calls.append(("pre", kwargs)) or _adjusted_daily_frame(),
    )
    monkeypatch.setattr(
        data_layer.pdd,
        "get_stock_daily",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("普通未复权接口不得用于协整价格")
        ),
    )

    result = data_layer.get_stock_daily_adjusted(
        symbols=["A"], start_date="20260105", end_date="20260106"
    )

    assert not result.empty
    assert [name for name, _ in calls] == ["pre"]


def test_adjusted_daily_supports_explicit_post_adjusted_basis(
        monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(data_layer, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_layer, "ensure_login", lambda: "ok")
    monkeypatch.setattr(
        data_layer.pdd,
        "get_stock_daily_post",
        lambda **kwargs: calls.append(("post", kwargs)) or _adjusted_daily_frame(),
    )

    result = data_layer.get_stock_daily_adjusted(
        symbols=["A"], start_date="20260105", end_date="20260106",
        price_basis="post_adjusted",
    )

    assert not result.empty
    assert [name for name, _ in calls] == ["post"]


def test_raw_price_basis_is_rejected_before_data_access(monkeypatch):
    monkeypatch.setattr(
        data_layer, "ensure_login",
        lambda: (_ for _ in ()).throw(AssertionError("不应尝试登录")),
    )

    with pytest.raises(ValueError, match="复权"):
        data_layer.get_stock_daily_adjusted(
            symbols=["A"], start_date="20260105", end_date="20260106",
            price_basis="raw",
        )


def test_price_panel_records_research_price_basis(monkeypatch):
    monkeypatch.setattr(
        data_layer,
        "get_stock_daily_adjusted",
        lambda **kwargs: _adjusted_daily_frame(),
    )

    panel = data_layer.build_price_panel(
        ["A"], "20260105", "20260106", price_basis="pre_adjusted"
    )

    assert panel.attrs["price_basis"] == "pre_adjusted"
    assert panel.attrs["source_method"] == "get_stock_daily_pre"


def test_login_rejects_http(monkeypatch):
    monkeypatch.setattr(data_layer, "_TOKEN", None)
    monkeypatch.setattr(data_layer, "_load_dotenv", lambda: None)
    monkeypatch.setenv("PANDADATA_USER", "test_user")
    monkeypatch.setenv("PANDADATA_PASSWORD", "test_password")
    monkeypatch.setenv(
        "PANDADATA_BASE_URL", "http://data.example.test"
    )
    monkeypatch.delenv("PANDADATA_TOKEN", raising=False)
    monkeypatch.setattr(
        data_layer.pdd,
        "init_token",
        lambda *args, **kwargs: "must-not-be-used",
    )

    with pytest.raises(
        RuntimeError, match="https://"
    ):
        data_layer.ensure_login()


def test_login_uses_memory_only_auth_and_token_only_data_client(
        tmp_path, monkeypatch):
    import panda_data.auth_manager as auth_manager
    import panda_data.client as panda_client

    calls = []
    monkeypatch.setattr(data_layer, "_TOKEN", None)
    monkeypatch.setattr(data_layer, "_load_dotenv", lambda: None)
    monkeypatch.setattr(auth_manager, "_user_json_dir", str(tmp_path))

    def fake_init_token(username="", password="", base_url=None, **kwargs):
        auth_manager.save_auth_state(
            username,
            password,
            base_url or "",
            "token-value",
            3600,
        )
        return "token-value"

    monkeypatch.setattr(data_layer.pdd, "init_token", fake_init_token)
    def fake_init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            config=SimpleNamespace(
                username="",
                password="",
                verify_ssl=kwargs["verify_ssl"],
            )
        )

    monkeypatch.setattr(panda_client, "init", fake_init)
    monkeypatch.setenv("PANDADATA_USER", "test_user")
    monkeypatch.setenv("PANDADATA_PASSWORD", "test_password")
    monkeypatch.setenv(
        "PANDADATA_BASE_URL", "https://data.example.test"
    )
    monkeypatch.delenv("PANDADATA_TOKEN", raising=False)

    assert data_layer.ensure_login() == "token-value"
    assert not (tmp_path / "user.json").exists()
    assert calls[-1]["username"] == ""
    assert calls[-1]["password"] == ""
    assert calls[-1]["verify_ssl"] is True
    assert auth_manager.get_username() == ""
    assert auth_manager.get_password() is None
    assert auth_manager.re_login() is False


def test_login_accepts_ephemeral_token_without_password(
        tmp_path, monkeypatch):
    import panda_data.auth_manager as auth_manager
    import panda_data.client as panda_client

    calls = []
    monkeypatch.setattr(data_layer, "_TOKEN", None)
    monkeypatch.setattr(data_layer, "_load_dotenv", lambda: None)
    monkeypatch.setattr(auth_manager, "_user_json_dir", str(tmp_path))
    monkeypatch.setattr(
        data_layer.pdd,
        "init_token",
        lambda *args, **kwargs: pytest.fail(
            "password login must not run"
        ),
    )
    def fake_init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            config=SimpleNamespace(
                username="",
                password="",
                verify_ssl=kwargs["verify_ssl"],
            )
        )

    monkeypatch.setattr(panda_client, "init", fake_init)
    monkeypatch.delenv("PANDADATA_USER", raising=False)
    monkeypatch.delenv("PANDADATA_PASSWORD", raising=False)
    monkeypatch.setenv("PANDADATA_TOKEN", "ephemeral-token")
    monkeypatch.setenv(
        "PANDADATA_BASE_URL", "https://data.example.test"
    )

    assert data_layer.ensure_login() == "ephemeral-token"
    assert not (tmp_path / "user.json").exists()
    assert calls[-1]["username"] == ""
    assert calls[-1]["password"] == ""
    assert calls[-1]["verify_ssl"] is True


def test_login_guards_ignore_existing_sdk_credential_file(
        tmp_path, monkeypatch):
    import panda_data.auth_manager as auth_manager
    import panda_data.readers.future_reader as future_reader
    import panda_data.transport.http as transport
    from pandadata_security import (
        PandaDataSecurityError,
        _RejectRedirects,
        _install_memory_only_guards,
    )
    from urllib.error import HTTPError
    from urllib.request import Request

    marker = '{"encrypted_credentials":"must-not-be-read"}'
    credential_file = tmp_path / "user.json"
    credential_file.write_text(marker, encoding="utf-8")
    legacy_clear_auth = auth_manager.clear_auth
    monkeypatch.setattr(auth_manager, "_user_json_dir", str(tmp_path))
    _install_memory_only_guards(
        auth_manager, transport, future_reader
    )

    assert auth_manager.get_decrypted_credentials() is None
    assert auth_manager.re_login() is False
    assert transport._auth_auto_login() is False
    assert transport._auth_re_login() is False
    legacy_clear_auth()
    assert credential_file.read_text(encoding="utf-8") == marker
    with pytest.raises(PandaDataSecurityError):
        future_reader.download_future_research_report(
            symbol="CU", date="20260710", pub="test"
        )
    client = transport.HTTPClient(transport.HTTPClientConfig(
        base_url="https://data.example.test"
    ))
    redirect_guard = next(
        item for item in client._opener.handlers
        if isinstance(item, _RejectRedirects)
    )
    for target in (
        "http://data.example.test/downgrade",
        "https://other.example.test/cross-origin",
    ):
        with pytest.raises(HTTPError):
            redirect_guard.redirect_request(
                Request("https://data.example.test/source"),
                None,
                302,
                "Found",
                {},
                target,
            )
    monkeypatch.setattr(
        client, "_get_token_file_path", lambda: str(credential_file)
    )
    client.close()
    assert credential_file.read_text(encoding="utf-8") == marker
