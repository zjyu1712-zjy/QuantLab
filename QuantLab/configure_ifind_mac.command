#!/bin/bash
set -euo pipefail

SERVICE="QuantLab-iFinD"
ACCOUNT="$(id -un)"
SECURITY_BIN="${QUANTLAB_SECURITY_BIN:-/usr/bin/security}"
SYSTEM_NAME="${QUANTLAB_UNAME_FOR_TESTS:-$(uname -s)}"

if [[ "$SYSTEM_NAME" != "Darwin" ]]; then
    echo "此配置命令仅适用于 macOS。" >&2
    exit 1
fi

echo "请在接下来的钥匙串提示中粘贴 iFinD refresh_token（输入不会显示）。"
if ! "$SECURITY_BIN" add-generic-password -U -a "$ACCOUNT" -s "$SERVICE" -w; then
    echo "未能保存 iFinD 凭证，请检查钥匙串权限。" >&2
    exit 1
fi

if ! stored_value="$("$SECURITY_BIN" find-generic-password -a "$ACCOUNT" -s "$SERVICE" -w 2>/dev/null)"; then
    echo "未能验证钥匙串记录，请检查钥匙串权限。" >&2
    exit 1
fi

if [[ -z "$stored_value" ]]; then
    "$SECURITY_BIN" delete-generic-password -a "$ACCOUNT" -s "$SERVICE" >/dev/null 2>&1 || true
    echo "refresh_token 不能为空，未保存配置。" >&2
    exit 1
fi

unset stored_value
echo "iFinD 凭证已安全保存到 macOS 钥匙串。"
