"""Fail-closed PandaData session setup.

panda_data 0.0.12 defaults to HTTP and persists recoverable credentials.
This adapter requires HTTPS, disables SDK disk-based re-login for the life
of the process, and reinitializes the data client with token-only
authentication.
"""
from __future__ import annotations

import importlib
import os
import tempfile
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler


DEFAULT_BASE_URL = ""
EMPTY_CREDENTIAL = ""


class PandaDataSecurityError(RuntimeError):
    """Raised when a PandaData session cannot meet the security contract."""


class _RejectRedirects(HTTPRedirectHandler):
    """Reject redirects so credentials and tokens cannot change origin."""

    def redirect_request(
        self, req, fp, code, msg, headers, newurl
    ):
        raise HTTPError(
            req.full_url,
            code,
            "PandaData 安全策略禁止 HTTP 重定向",
            headers,
            fp,
        )


def _base_url() -> str:
    return (
        os.environ.get("PANDADATA_BASE_URL")
        or os.environ.get("JAVA_SERVICE_BASE_URL")
        or DEFAULT_BASE_URL
    ).strip().rstrip("/")


def _validate_transport(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise PandaDataSecurityError(
            "PANDADATA_BASE_URL 必须是完整的 https:// 地址；"
            "为避免凭证或令牌被窃听，本工具不支持 HTTP"
        )
    if parsed.username or parsed.password:
        raise PandaDataSecurityError(
            "PANDADATA_BASE_URL 不得包含账号或密码"
        )
    return base_url


def _sdk_modules() -> tuple[Any, Any, Any, Any, Any]:
    try:
        auth_manager = importlib.import_module("panda_data.auth_manager")
        client_module = importlib.import_module("panda_data.client")
        token_module = importlib.import_module(
            "panda_data.readers.init_token"
        )
        transport_module = importlib.import_module(
            "panda_data.transport.http"
        )
        future_module = importlib.import_module(
            "panda_data.readers.future_reader"
        )
    except Exception as exc:
        raise PandaDataSecurityError(
            "当前 panda_data 安装不支持安全会话适配"
        ) from exc

    if not callable(getattr(auth_manager, "save_auth_state", None)):
        raise PandaDataSecurityError(
            "当前 panda_data 缺少内存令牌接口"
        )
    if not callable(getattr(auth_manager, "_persist_credentials", None)):
        raise PandaDataSecurityError(
            "当前 panda_data 无法禁用凭证落盘"
        )
    if not callable(
        getattr(auth_manager, "_read_persisted_credentials", None)
    ):
        raise PandaDataSecurityError(
            "当前 panda_data 无法禁用磁盘凭证读取"
        )
    if not callable(getattr(client_module, "init", None)):
        raise PandaDataSecurityError(
            "当前 panda_data 缺少数据客户端初始化接口"
        )
    if not hasattr(token_module, "init_client"):
        raise PandaDataSecurityError(
            "当前 panda_data 无法隔离带密码的数据客户端"
        )
    for name in ("_auth_auto_login", "_auth_re_login"):
        if not callable(getattr(transport_module, name, None)):
            raise PandaDataSecurityError(
                "当前 panda_data 无法禁用磁盘自动重登录"
            )
    if not callable(
        getattr(transport_module.HTTPClient, "_delete_token_file", None)
    ):
        raise PandaDataSecurityError(
            "当前 panda_data 无法阻止删除既有磁盘凭证"
        )
    if not callable(
        getattr(future_module, "download_future_research_report", None)
    ):
        raise PandaDataSecurityError(
            "当前 panda_data 无法隔离研报下载凭证路径"
        )
    return (
        auth_manager,
        client_module,
        token_module,
        transport_module,
        future_module,
    )


def _install_memory_only_guards(
        auth_manager: Any,
        transport_module: Any,
        future_module: Any,
) -> None:
    """Disable credential persistence and disk re-login process-wide."""
    def _do_not_persist(*args, **kwargs) -> None:
        return None

    def _do_not_read(*args, **kwargs) -> None:
        return None

    def _do_not_relogin(*args, **kwargs) -> bool:
        return False

    def _do_not_delete(*args, **kwargs) -> None:
        return None

    def _disabled_disk_feature(*args, **kwargs):
        raise PandaDataSecurityError(
            "当前安全会话禁用依赖 user.json 的第三方研报下载"
        )

    isolated_path = os.path.join(
        tempfile.gettempdir(),
        f"quantskills-pandadata-disabled-{os.getpid()}",
        "user.json",
    )

    current_build_opener = transport_module.build_opener
    if not getattr(current_build_opener, "_quantskills_guarded", False):
        def _secure_build_opener(*handlers):
            return current_build_opener(
                _RejectRedirects(), *handlers
            )

        _secure_build_opener._quantskills_guarded = True
        transport_module.build_opener = _secure_build_opener

    auth_manager._persist_credentials = _do_not_persist
    auth_manager._read_persisted_credentials = _do_not_read
    auth_manager._get_user_json_path = lambda: isolated_path
    auth_manager.auto_login_from_disk = _do_not_relogin
    auth_manager.re_login = _do_not_relogin
    transport_module._auth_auto_login = _do_not_relogin
    transport_module._auth_re_login = _do_not_relogin
    transport_module.HTTPClient._delete_token_file = _do_not_delete
    future_module.download_future_research_report = (
        _disabled_disk_feature
    )


def establish_session(
    panda_data_module: Any,
    *,
    username: str = "",
    password: str = "",
    token: str = "",
) -> str:
    """Create a memory-only, token-authenticated PandaData session."""
    base_url = _validate_transport(_base_url())
    username = username.strip()
    password = password.strip()
    token = token.strip()
    if not token and (not username or not password):
        raise PandaDataSecurityError(
            "缺少 PandaData 临时令牌或账号密码"
        )

    (
        auth_manager,
        client_module,
        token_module,
        transport_module,
        future_module,
    ) = _sdk_modules()
    _install_memory_only_guards(
        auth_manager, transport_module, future_module
    )
    original_init_client = token_module.init_client

    def _do_not_persist(*args, **kwargs) -> None:
        return None

    token_module.init_client = _do_not_persist
    try:
        if token:
            session_token = token
            expires_in = 14400
        else:
            session_token = panda_data_module.init_token(
                username=username,
                password=password,
                base_url=base_url,
            )
            expires_in = int(auth_manager.get_token_ttl())
        if not session_token:
            raise PandaDataSecurityError(
                "PandaData 未返回有效令牌"
            )
        auth_manager.save_auth_state(
            username=EMPTY_CREDENTIAL,
            password=EMPTY_CREDENTIAL,
            base_url=base_url,
            token=session_token,
            expires_in=expires_in,
        )

        config_module = importlib.import_module("panda_data.config")
        sdk_config = config_module.get_config()
        sdk_config["DEFAULT_USERNAME"] = EMPTY_CREDENTIAL
        sdk_config["DEFAULT_PASSWORD"] = EMPTY_CREDENTIAL
        client = client_module.init(
            username=EMPTY_CREDENTIAL,
            password=EMPTY_CREDENTIAL,
            base_url=base_url,
            verify_ssl=True,
        )
        client_config = getattr(client, "config", None)
        if (
            getattr(client_config, "username", "")
            or getattr(client_config, "password", "")
            or getattr(client_config, "verify_ssl", True) is not True
        ):
            raise PandaDataSecurityError(
                "PandaData 数据客户端未满足令牌与证书校验要求"
            )
        return str(session_token)
    finally:
        token_module.init_client = original_init_client
