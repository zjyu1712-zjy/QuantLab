from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURE_SCRIPT = PROJECT_ROOT / "configure_ifind_mac.command"


def _write_fake_security(tmp_path: Path) -> tuple[Path, Path]:
    log_path = tmp_path / "security-arguments.log"
    fake_security = tmp_path / "security"
    fake_security.write_text(
        """#!/bin/bash
set -u
command_name="$1"
printf '%s' "$1" >> "$FAKE_SECURITY_LOG"
shift
for argument in "$@"; do
    printf '\\t%s' "$argument" >> "$FAKE_SECURITY_LOG"
done
printf '\\n' >> "$FAKE_SECURITY_LOG"

case "$command_name" in
    add-generic-password)
        exit "${FAKE_SECURITY_ADD_EXIT_CODE:-0}"
        ;;
    find-generic-password)
        if [[ "${FAKE_SECURITY_FIND_EXIT_CODE:-0}" != "0" ]]; then
            exit "$FAKE_SECURITY_FIND_EXIT_CODE"
        fi
        printf '%s' "${FAKE_SECURITY_TOKEN:-}"
        ;;
    delete-generic-password)
        exit 0
        ;;
esac
""",
        encoding="utf-8",
    )
    fake_security.chmod(0o700)
    return fake_security, log_path


def _run_configure(
    tmp_path: Path,
    *,
    stored_token: str = "fake-refresh-token",
    system_name: str = "Darwin",
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    fake_security, log_path = _write_fake_security(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_SECURITY_LOG": str(log_path),
            "FAKE_SECURITY_TOKEN": stored_token,
            "QUANTLAB_SECURITY_BIN": str(fake_security),
            "QUANTLAB_UNAME_FOR_TESTS": system_name,
        }
    )
    result = subprocess.run(
        ["/bin/bash", str(CONFIGURE_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    recorded_commands = (
        log_path.read_text(encoding="utf-8").splitlines()
        if log_path.exists()
        else []
    )
    return result, recorded_commands


def test_configure_uses_keychain_prompt_without_secret_in_argv_or_output(
    tmp_path: Path,
) -> None:
    assert CONFIGURE_SCRIPT.exists(), "配置脚本尚未实现"

    result, recorded_commands = _run_configure(tmp_path)

    assert result.returncode == 0
    assert recorded_commands[0].split("\t")[-1] == "-w"
    assert "fake-refresh-token" not in "\n".join(recorded_commands)
    assert "fake-refresh-token" not in result.stdout
    assert "fake-refresh-token" not in result.stderr


def test_configure_rejects_empty_keychain_value_and_removes_it(
    tmp_path: Path,
) -> None:
    assert CONFIGURE_SCRIPT.exists(), "配置脚本尚未实现"

    result, recorded_commands = _run_configure(tmp_path, stored_token="")

    assert result.returncode != 0
    assert any(command.startswith("delete-generic-password\t") for command in recorded_commands)
    assert "不能为空" in result.stderr


def test_configure_rejects_non_macos(tmp_path: Path) -> None:
    assert CONFIGURE_SCRIPT.exists(), "配置脚本尚未实现"

    result, recorded_commands = _run_configure(tmp_path, system_name="Linux")

    assert result.returncode != 0
    assert recorded_commands == []
    assert "仅适用于 macOS" in result.stderr
