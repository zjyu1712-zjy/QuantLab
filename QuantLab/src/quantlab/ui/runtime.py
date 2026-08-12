"""Streamlit 本地模式与临时公开模式的运行时边界。"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st


def is_public_mode() -> bool:
    """公开分享模式只在显式环境变量启用时生效。"""
    return os.environ.get("QUANTLAB_PUBLIC_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def require_access() -> None:
    """公开分享时要求访问口令，本地模式保持原有使用体验。"""
    expected_password = os.environ.get("QUANTLAB_ACCESS_PASSWORD", "")
    if not expected_password:
        return

    if st.session_state.get("_quantlab_authenticated") is True:
        return

    st.title("QuantLab 访问验证")
    st.caption("请输入页面所有者提供的临时访问口令。")
    supplied_password = st.text_input(
        "访问口令",
        type="password",
        key="_quantlab_access_password",
    )
    if st.button("进入 QuantLab", type="primary"):
        if hmac.compare_digest(supplied_password, expected_password):
            st.session_state["_quantlab_authenticated"] = True
            st.rerun()
        else:
            st.error("访问口令不正确")
    st.info(
        "公开分享模式不会连接券商或自动下单；请勿上传身份证号、"
        "银行卡号等与量化研究无关的敏感信息。"
    )
    st.stop()


def get_market_storage_dirs(project_root: str | Path) -> tuple[Path, Path]:
    """返回本地持久目录，或公开模式下每个会话的独立临时目录。"""
    root = Path(project_root)
    if not is_public_mode():
        return (
            root / "data" / "raw" / "market",
            root / "data" / "processed" / "market",
        )

    temporary_directory = st.session_state.get(
        "_quantlab_public_temporary_directory"
    )
    if temporary_directory is None:
        temporary_directory = TemporaryDirectory(
            prefix="quantlab-public-session-"
        )
        st.session_state[
            "_quantlab_public_temporary_directory"
        ] = temporary_directory

    session_root = Path(temporary_directory.name)
    return (
        session_root / "raw" / "market",
        session_root / "processed" / "market",
    )
