#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE="QuantLab-iFinD"
ACCOUNT="$(id -un)"
SECURITY_BIN="${QUANTLAB_SECURITY_BIN:-/usr/bin/security}"
SYSTEM_NAME="${QUANTLAB_UNAME_FOR_TESTS:-$(uname -s)}"
PYTHON_BIN="${QUANTLAB_PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"

if [[ "$SYSTEM_NAME" != "Darwin" ]]; then
    echo "此启动命令仅适用于 macOS。" >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "未找到 QuantLab 虚拟环境，请先重建 .venv。" >&2
    exit 1
fi

if ! token="$("$SECURITY_BIN" find-generic-password -a "$ACCOUNT" -s "$SERVICE" -w 2>/dev/null)" || [[ -z "$token" ]]; then
    echo "尚未找到 iFinD 凭证，请先运行 ./configure_ifind_mac.command。" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
export IFIND_REFRESH_TOKEN="$token"
export MARKET_DATA_PROVIDER="ifind"
export PYTHONPATH="src"
unset token

exec "$PYTHON_BIN" -m streamlit run app.py
