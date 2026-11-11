from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURE_SCRIPT = PROJECT_ROOT / "configure_ifind_mac.command"
RUN_SCRIPT = PROJECT_ROOT / "run_ifind_mac.command"


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
    stored_token: str = "fake-access-token",
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
    assert any("QuantLab-iFinD-AccessToken" in command for command in recorded_commands)
    assert "fake-access-token" not in "\n".join(recorded_commands)
    assert "fake-access-token" not in result.stdout
    assert "fake-access-token" not in result.stderr


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


def _write_fake_python(tmp_path: Path) -> tuple[Path, Path]:
    state_path = tmp_path / "python-child-state.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/bin/bash
set -u
if [[ "${IFIND_ACCESS_TOKEN:-}" == "${EXPECTED_FAKE_TOKEN:-}" ]]; then
    printf 'token_matches=true\\n' > "$FAKE_PYTHON_STATE"
else
    printf 'token_matches=false\\n' > "$FAKE_PYTHON_STATE"
fi
printf 'provider=%s\\n' "${MARKET_DATA_PROVIDER:-}" >> "$FAKE_PYTHON_STATE"
printf 'pythonpath=%s\\n' "${PYTHONPATH:-}" >> "$FAKE_PYTHON_STATE"
printf 'refresh_token_set=%s\\n' "${IFIND_REFRESH_TOKEN:+true}" >> "$FAKE_PYTHON_STATE"
printf 'argv=' >> "$FAKE_PYTHON_STATE"
printf '%s\\t' "$@" >> "$FAKE_PYTHON_STATE"
printf '\\n' >> "$FAKE_PYTHON_STATE"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    return fake_python, state_path


def _run_launcher(
    tmp_path: Path,
    *,
    fake_token: str = "fake-access-token",
    keychain_exit_code: int = 0,
    create_project_python: bool = True,
    system_name: str = "Darwin",
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    fake_security, security_log_path = _write_fake_security(tmp_path)
    fake_python, state_path = _write_fake_python(tmp_path)
    if not create_project_python:
        fake_python.unlink()

    environment = os.environ.copy()
    environment.update(
        {
            "EXPECTED_FAKE_TOKEN": fake_token,
            "FAKE_PYTHON_STATE": str(state_path),
            "FAKE_SECURITY_FIND_EXIT_CODE": str(keychain_exit_code),
            "FAKE_SECURITY_LOG": str(security_log_path),
            "FAKE_SECURITY_TOKEN": fake_token,
            "QUANTLAB_PYTHON_BIN": str(fake_python),
            "QUANTLAB_SECURITY_BIN": str(fake_security),
            "QUANTLAB_UNAME_FOR_TESTS": system_name,
            "IFIND_REFRESH_TOKEN": "legacy-refresh-token",
        }
    )
    result = subprocess.run(
        ["/bin/bash", str(RUN_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    state: dict[str, str] = {}
    if state_path.exists():
        for line in state_path.read_text(encoding="utf-8").splitlines():
            key, value = line.split("=", maxsplit=1)
            state[key] = value
    return result, state


def test_launcher_sets_ifind_child_environment_without_printing_token(
    tmp_path: Path,
) -> None:
    assert RUN_SCRIPT.exists(), "启动脚本尚未实现"

    result, child_state = _run_launcher(tmp_path)

    assert result.returncode == 0
    assert child_state == {
        "token_matches": "true",
        "provider": "ifind",
        "pythonpath": "src",
        "refresh_token_set": "",
        "argv": "-m\tstreamlit\trun\tapp.py\t",
    }
    assert "fake-access-token" not in result.stdout
    assert "fake-access-token" not in result.stderr


def test_launcher_explains_missing_keychain_credential(tmp_path: Path) -> None:
    assert RUN_SCRIPT.exists(), "启动脚本尚未实现"

    result, child_state = _run_launcher(tmp_path, keychain_exit_code=44)

    assert result.returncode != 0
    assert child_state == {}
    assert "configure_ifind_mac.command" in result.stderr


def test_launcher_explains_missing_virtual_environment(tmp_path: Path) -> None:
    assert RUN_SCRIPT.exists(), "启动脚本尚未实现"

    result, child_state = _run_launcher(tmp_path, create_project_python=False)

    assert result.returncode != 0
    assert child_state == {}
    assert ".venv" in result.stderr


def test_launcher_rejects_non_macos(tmp_path: Path) -> None:
    assert RUN_SCRIPT.exists(), "启动脚本尚未实现"

    result, child_state = _run_launcher(tmp_path, system_name="Linux")

    assert result.returncode != 0
    assert child_state == {}
    assert "仅适用于 macOS" in result.stderr
