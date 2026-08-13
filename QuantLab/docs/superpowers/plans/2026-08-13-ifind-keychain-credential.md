# iFinD Keychain Credential Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe macOS Keychain configuration command and a one-command QuantLab launcher that restores the iFinD credential after terminal or computer restarts.

**Architecture:** Keep credential storage outside Python and outside the repository. A configuration script delegates hidden token entry to macOS `security`, while a launch script reads the generic-password item only at runtime and injects it into the Streamlit process environment. Shell-level tests use fake `security` and Python executables, so automated verification never touches the real Keychain or network.

**Tech Stack:** POSIX-compatible zsh/bash scripting on macOS, macOS `/usr/bin/security`, Python `subprocess`/pytest, Streamlit, Git.

## Global Constraints

- The Keychain service is exactly `QuantLab-iFinD`.
- The Keychain account is the current macOS username.
- The real `IFIND_REFRESH_TOKEN` must never enter source files, `.env` files, logs, test output, Git history, or command-line arguments.
- Only the launched Streamlit process receives `IFIND_REFRESH_TOKEN` and `MARKET_DATA_PROVIDER=ifind`.
- Existing iFinD Provider, market-data service, analysis, and backtest modules must not change.
- Tests must not access the real Keychain or external network.
- Website deployment remains out of scope and will use server-side secret management rather than macOS Keychain.

---

### Task 1: Keychain configuration command

**Files:**
- Create: `configure_ifind_mac.command`
- Create: `tests/test_ifind_keychain_scripts.py`

**Interfaces:**
- Consumes: macOS `security add-generic-password -U -a <account> -s QuantLab-iFinD -w`, where the final `-w` makes `security` prompt without receiving the token in argv.
- Produces: a generic-password item readable with `security find-generic-password -a <account> -s QuantLab-iFinD -w`.

- [ ] **Step 1: Write failing configuration-script tests**

Create pytest helpers that copy the command into `tmp_path`, inject a fake executable through `QUANTLAB_SECURITY_BIN`, and capture stdout/stderr and the fake command argv. Add these cases:

```python
def test_configure_uses_keychain_prompt_without_secret_in_argv_or_output(
    tmp_path: Path,
) -> None:
    result, recorded_argv = run_configure(
        tmp_path,
        stored_token="fake-refresh-token",
    )
    assert result.returncode == 0
    assert recorded_argv[-1] == "-w"
    assert "fake-refresh-token" not in recorded_argv
    assert "fake-refresh-token" not in result.stdout


def test_configure_rejects_empty_keychain_value_and_removes_it(tmp_path: Path) -> None:
    result, recorded_commands = run_configure(tmp_path, stored_token="")
    assert result.returncode != 0
    assert "delete-generic-password" in recorded_commands


def test_configure_rejects_non_macos(tmp_path: Path) -> None:
    result, _ = run_configure(tmp_path, system_name="Linux")
    assert result.returncode != 0
    assert "仅适用于 macOS" in result.stderr
```

`run_configure` must build a temporary fake `security` executable, set
`QUANTLAB_SECURITY_BIN` and `QUANTLAB_UNAME_FOR_TESTS`, execute the repository
script with `subprocess.run([str(CONFIGURE_SCRIPT)], env=environment,
text=True, capture_output=True)`, and return
the completed process plus the fake executable's newline-delimited argv log.

- [ ] **Step 2: Run the focused tests and verify red**

Run:

```bash
cd /Users/zhoujiayu/Documents/量化/QuantLab
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ifind_keychain_scripts.py -v
```

Expected: FAIL because `configure_ifind_mac.command` does not exist.

- [ ] **Step 3: Implement the minimal configuration command**

Implement a script with this control flow:

```bash
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
"$SECURITY_BIN" add-generic-password -U -a "$ACCOUNT" -s "$SERVICE" -w

stored_value="$("$SECURITY_BIN" find-generic-password -a "$ACCOUNT" -s "$SERVICE" -w 2>/dev/null)" || {
    echo "未能验证钥匙串记录，请检查钥匙串权限。" >&2
    exit 1
}
if [[ -z "$stored_value" ]]; then
    "$SECURITY_BIN" delete-generic-password -a "$ACCOUNT" -s "$SERVICE" >/dev/null 2>&1 || true
    echo "refresh_token 不能为空，未保存配置。" >&2
    exit 1
fi
unset stored_value
echo "iFinD 凭证已安全保存到 macOS 钥匙串。"
```

The production path is fixed to `/usr/bin/security`; the override exists only so tests can substitute a fake executable. Do not add `-A`, because that permits unrestricted application access.

- [ ] **Step 4: Run focused tests and shell syntax check**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ifind_keychain_scripts.py -v
bash -n configure_ifind_mac.command
```

Expected: configuration cases PASS and syntax check exits 0.

- [ ] **Step 5: Commit Task 1**

```bash
git add QuantLab/configure_ifind_mac.command QuantLab/tests/test_ifind_keychain_scripts.py
git commit -m "feat: store iFinD credential in macOS Keychain"
```

---

### Task 2: Keychain-backed QuantLab launcher

**Files:**
- Create: `run_ifind_mac.command`
- Modify: `tests/test_ifind_keychain_scripts.py`

**Interfaces:**
- Consumes: the Keychain item created by Task 1 and project-local `.venv/bin/python`.
- Produces: a Streamlit process launched as `.venv/bin/python -m streamlit run app.py` with `PYTHONPATH=src`, `MARKET_DATA_PROVIDER=ifind`, and the runtime-only `IFIND_REFRESH_TOKEN`.

- [ ] **Step 1: Write failing launcher tests**

Extend the same isolated fake-command harness with these cases:

```python
def test_launcher_reads_keychain_and_sets_child_environment_without_printing_token(
    tmp_path: Path,
) -> None:
    result, child_environment, child_arguments = run_launcher(
        tmp_path,
        fake_token="fake-refresh-token",
    )
    assert result.returncode == 0
    assert child_environment["IFIND_REFRESH_TOKEN"] == "fake-refresh-token"
    assert child_environment["MARKET_DATA_PROVIDER"] == "ifind"
    assert child_environment["PYTHONPATH"] == "src"
    assert child_arguments == ["-m", "streamlit", "run", "app.py"]
    assert "fake-refresh-token" not in result.stdout
    assert "fake-refresh-token" not in result.stderr


def test_launcher_explains_missing_keychain_credential(tmp_path: Path) -> None:
    result, _, _ = run_launcher(tmp_path, keychain_exit_code=44)
    assert result.returncode != 0
    assert "configure_ifind_mac.command" in result.stderr


def test_launcher_explains_missing_virtual_environment(tmp_path: Path) -> None:
    result, _, _ = run_launcher(tmp_path, create_project_python=False)
    assert result.returncode != 0
    assert ".venv" in result.stderr
```

`run_launcher` must copy the launcher into an isolated temporary project,
create a fake `security` executable and (unless disabled) a fake project Python
executable, then set `QUANTLAB_SECURITY_BIN`, `QUANTLAB_PYTHON_BIN`, and
`QUANTLAB_UNAME_FOR_TESTS=Darwin`. The fake Python executable writes only its
selected non-secret environment values and argv to files for assertions; it
must never write the token value to a diagnostic file.

- [ ] **Step 2: Run launcher tests and verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ifind_keychain_scripts.py -v
```

Expected: launcher cases FAIL because `run_ifind_mac.command` does not exist.

- [ ] **Step 3: Implement the minimal launcher**

Implement a script with this control flow:

```bash
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
```

The launcher must never use `set -x`, print environment variables, persist shell exports, or write a token-bearing file.

- [ ] **Step 4: Run focused tests and syntax checks**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ifind_keychain_scripts.py -v
bash -n configure_ifind_mac.command
bash -n run_ifind_mac.command
```

Expected: all script tests PASS and both syntax checks exit 0.

- [ ] **Step 5: Commit Task 2**

```bash
git add QuantLab/run_ifind_mac.command QuantLab/tests/test_ifind_keychain_scripts.py
git commit -m "feat: launch QuantLab with Keychain credential"
```

---

### Task 3: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_ifind_keychain_scripts.py`

**Interfaces:**
- Consumes: `./configure_ifind_mac.command` and `./run_ifind_mac.command` from Tasks 1–2.
- Produces: a beginner-facing setup and restart workflow that does not recommend plaintext secrets.

- [ ] **Step 1: Add a failing documentation assertion**

Add a test that reads `README.md` and asserts that it contains both command names, states that Keychain is used, and does not recommend storing a real token in `.env` or shell profile files:

```python
def test_readme_documents_keychain_workflow():
    text = README_PATH.read_text(encoding="utf-8")
    assert "./configure_ifind_mac.command" in text
    assert "./run_ifind_mac.command" in text
    assert "macOS 钥匙串" in text
```

- [ ] **Step 2: Run the documentation test and verify red**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ifind_keychain_scripts.py::test_readme_documents_keychain_workflow -v
```

Expected: FAIL because the two new commands are not documented.

- [ ] **Step 3: Update README with exact user workflow**

Document:

```text
首次配置：双击或运行 ./configure_ifind_mac.command，并在钥匙串提示中粘贴 refresh_token。
以后启动：双击或运行 ./run_ifind_mac.command。
更换 token：重新运行配置命令，它会更新同一条钥匙串记录。
公开网站：使用服务器 Secret Manager，且必须先确认 iFinD 多用户授权；不要复制本机钥匙串方案到服务器。
```

Remove no existing provider-mode, authorization, or integration-test documentation.

- [ ] **Step 4: Run complete verification**

Run:

```bash
bash -n configure_ifind_mac.command
bash -n run_ifind_mac.command
PYTHONPATH=src .venv/bin/python -m pytest -v
git diff --check
git status --short
```

Expected: both syntax checks exit 0; full offline suite passes with live integration tests skipped; diff check is clean; only intended files are modified.

- [ ] **Step 5: Commit Task 3**

```bash
git add QuantLab/README.md QuantLab/tests/test_ifind_keychain_scripts.py
git commit -m "docs: explain secure iFinD startup"
```

- [ ] **Step 6: Manual handoff without reading the real token**

Ask the user to run:

```bash
cd "/Users/zhoujiayu/Documents/量化/QuantLab"
./configure_ifind_mac.command
./run_ifind_mac.command
```

Success means a fresh terminal can start QuantLab and an iFinD market-data request succeeds without manually exporting the token. Do not run the configuration command on the user's behalf because token entry and Keychain authorization require their direct interaction.
