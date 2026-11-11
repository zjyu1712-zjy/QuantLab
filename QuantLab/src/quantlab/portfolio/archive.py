"""显式保存、列出和读取组合实验；不覆盖、不自动重算、不依赖原行情目录。"""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from tempfile import NamedTemporaryFile

from quantlab.backtest import provenance
from quantlab.portfolio.archive_codec import (
    canonical_id, decode_analysis, encode_analysis, iso_date, keys, require, utc_time,
    validate_pair, verify_current_performance,
)
from quantlab.portfolio.types import PortfolioAnalysis, PortfolioBacktestError


SCHEMA_VERSION = "portfolio-archive-v1"


@dataclass(frozen=True)
class SavedPortfolioRun:
    run_id: str
    name: str
    saved_at: str
    path: Path
    benchmark_status: str


@dataclass(frozen=True)
class LoadedPortfolioRun:
    record: SavedPortfolioRun
    strategy: PortfolioAnalysis
    benchmark: PortfolioAnalysis | None
    benchmark_status: str
    benchmark_failure: dict | None
    warnings: tuple[str, ...]
    # 保存同一次读取、校验的原始快照；不在 repr 中打印完整研究数据。
    archive_bytes: bytes = field(default=b"", repr=False, compare=False)


@dataclass(frozen=True)
class PortfolioRunScan:
    records: tuple[SavedPortfolioRun, ...]
    issues: tuple[str, ...]


def _json(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _checksum(payload):
    return sha256(_json(payload).encode("utf-8")).hexdigest()


def _name(value):
    require(isinstance(value, str) and bool(value.strip()) and len(value.strip()) <= 200,
            "实验名称需要 1–200 个字符")
    return value.strip()


def _root(directory):
    try:
        return Path(directory).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("组合档案目录无效") from error


def _failure(error):
    """只保存有限的结构化上下文，绝不把异常全文、凭证或路径写进档案。"""
    if error is None:
        return None
    require(isinstance(error, Exception), "基准失败信息必须是异常对象")
    reason = getattr(error, "reason", None)
    allowed = {"missing_open", "missing_close", "missing_execution_price", "zero_volume", "invalid_volume",
               "numeric_range", "non_positive_equity", "invalid_config", "invalid_analytics_config",
               "invalid_research_input", "invalid_factor", "invalid_run_identity", "negative_cash",
               "negative_sell_proceeds", "invalid_market_data", "invalid_schedule"}
    failure = {"reason": reason if isinstance(reason, str) and reason in allowed else "unavailable", "date": None, "symbols": []}
    if isinstance(error, PortfolioBacktestError):
        if type(error.date) is date:
            failure["date"] = iso_date(error.date.isoformat())
        if isinstance(error.symbols, tuple):
            failure["symbols"] = sorted(set(s for s in error.symbols if isinstance(s, str) and re.fullmatch(r"[0-9]{6}", s)))
    return failure


def _validate_failure(value):
    keys(value, ("reason", "date", "symbols"))
    require(isinstance(value["reason"], str) and re.fullmatch(r"[a-z_]+", value["reason"]) is not None,
            "基准失败原因格式无效")
    if value["date"] is not None:
        iso_date(value["date"])
    require(isinstance(value["symbols"], list)
            and all(isinstance(s, str) and re.fullmatch(r"[0-9]{6}", s) for s in value["symbols"]), "失败上下文代码无效")
    require(value["symbols"] == sorted(set(value["symbols"])), "失败上下文代码重复或未排序")


def save_portfolio_run(
    strategy: PortfolioAnalysis, output_dir: Path, *, name: str,
    benchmark: PortfolioAnalysis | None = None, benchmark_error: Exception | None = None,
) -> SavedPortfolioRun:
    """保存当前完整分析，以策略原 run_id 命名；同一次运行只允许保存一次。

    先完成全部验证，再写临时文件，最后原子发布。旧文件和符号链接均不覆盖。
    benchmark=None 且无异常表示未运行；传入异常表示基准不可用但策略仍完整。
    """
    require(benchmark is None or benchmark_error is None, "成功基准和失败原因不能同时提供")
    name = _name(name)
    strategy = deepcopy(strategy)
    benchmark = deepcopy(benchmark)
    strategy_payload = encode_analysis(strategy)
    benchmark_payload = encode_analysis(benchmark, benchmark=True) if benchmark is not None else None
    if benchmark is not None:
        validate_pair(strategy, benchmark)
    # 只有当前已加载代码产生的结果才能用当前公式做保存前核验。
    current_code = provenance._LOADED_CODE_SHA256
    require(provenance.source_tree_sha256(provenance._SOURCE_ROOT) == current_code,
            "源码已更改，请重启进程并重新运行后再保存；已存档历史仍可读取")
    for analysis in (strategy,) if benchmark is None else (strategy, benchmark):
        require(analysis.backtest.metadata["code_sha256"] == current_code,
                "结果来自不同代码版本，不能使用当前公式重新验证后另存；请打开原历史档案")
        verify_current_performance(analysis)
    failure = _failure(benchmark_error)
    status = "complete" if benchmark is not None else ("unavailable" if failure is not None else "not_run")
    run_id = canonical_id(strategy.backtest.metadata["run_id"])
    saved_at = datetime.now(timezone.utc).isoformat()
    payload = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "name": name, "saved_at": saved_at,
               "strategy": strategy_payload, "benchmark": benchmark_payload, "benchmark_status": status,
               "benchmark_failure": failure}
    payload["content_sha256"] = _checksum(payload)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    root = _root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{run_id}.json"
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".portfolio-", suffix=".tmp", dir=root,
                                delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # link 不会覆盖已存在的目的项，也不跟随目的符号链接。
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return SavedPortfolioRun(run_id, name, saved_at, destination, status)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "档案包含重复 JSON 字段")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError("档案不能含 NaN 或 Infinity 常量")


def _read_archive_bytes(path):
    # 固定文件名、拒绝链接，open 时再拒绝链接；只读普通文件，避免 FIFO 阻塞。
    require(not path.is_symlink() and path.is_file(), "档案不存在或不是普通文件")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "档案必须是普通文件")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            return stream.read()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def load_portfolio_run(output_dir: Path, run_id: str) -> LoadedPortfolioRun:
    """读取保存值并检查 v1 结构和一致性；不调用当前因子、回测或绩效公式。"""
    run_id = canonical_id(run_id)
    path = _root(output_dir) / f"{run_id}.json"
    try:
        content = _read_archive_bytes(path)
        return _decode_archive_bytes(content, run_id=run_id, path=path)
    except OSError as error:
        raise ValueError("组合档案读取失败：文件不可读或结构不符合规范") from error


def _decode_archive_bytes(content: bytes, *, run_id: str, path: Path) -> LoadedPortfolioRun:
    """共用的纯内存解码入口；导出复用它，不再打开可能被替换的原文件。"""
    canonical_id(run_id)
    require(type(content) is bytes and bool(content), "档案原始字节缺失")
    try:
        payload = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_bad_constant)
        keys(payload, ("schema_version", "run_id", "name", "saved_at", "strategy", "benchmark",
                       "benchmark_status", "benchmark_failure", "content_sha256"))
        require(payload["schema_version"] == SCHEMA_VERSION, "不支持的组合档案版本")
        require(payload["run_id"] == run_id, "档案编号与文件名不一致")
        require(payload["name"] == _name(payload["name"]), "实验名称不规范")
        utc_time(payload["saved_at"])
        core = {key: value for key, value in payload.items() if key != "content_sha256"}
        require(payload["content_sha256"] == _checksum(core), "档案内容指纹不一致，拒绝读取损坏记录")
        strategy = decode_analysis(payload["strategy"])
        require(strategy.backtest.metadata["run_id"] == run_id, "策略运行编号与档案编号不一致")
        status, failure = payload["benchmark_status"], payload["benchmark_failure"]
        benchmark = None
        if status == "complete":
            require(failure is None and payload["benchmark"] is not None, "基准状态与内容不一致")
            benchmark = decode_analysis(payload["benchmark"], benchmark=True)
            validate_pair(strategy, benchmark)
        elif status in ("not_run", "unavailable"):
            require(payload["benchmark"] is None, "不可用基准不能携带成功结果")
            if status == "not_run":
                require(failure is None, "未运行基准不能携带失败原因")
            else:
                _validate_failure(failure)
        else:
            raise ValueError("不支持的基准状态")
        warnings = []
        if any(item.backtest.metadata["code_sha256"] != provenance._LOADED_CODE_SHA256
               for item in (strategy, benchmark) if item is not None):
            warnings.append("档案运行代码与当前进程版本不同；展示的是保存值，未重新回测或重算指标。")
        record = SavedPortfolioRun(run_id, payload["name"], payload["saved_at"], path, status)
        return LoadedPortfolioRun(record, strategy, benchmark, status, deepcopy(failure), tuple(warnings), content)
    except (OSError, UnicodeError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise ValueError("组合档案读取失败：文件不可读或结构不符合规范") from error


def list_portfolio_runs(output_dir: Path) -> PortfolioRunScan:
    """新到旧列出可读档案；坏文件单独报告，不导致好记录消失，不创建目录。"""
    root = _root(output_dir)
    if not root.exists():
        return PortfolioRunScan((), ())
    require(root.is_dir(), "组合档案路径必须是目录")
    records, issues = [], []
    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or path.suffix != ".json":
            continue
        try:
            record = load_portfolio_run(root, path.stem).record
        except ValueError:
            issues.append(f"{path.name}：档案不可读、格式不支持或一致性检查失败；未修改原文件。")
        else:
            records.append(record)
    records.sort(key=lambda record: (utc_time(record.saved_at), record.run_id), reverse=True)
    return PortfolioRunScan(tuple(records), tuple(issues))
