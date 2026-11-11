"""回测运行身份与输入指纹；不读取凭证或依赖 Git 工作区是否干净。"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from uuid import uuid4

import pandas as pd

from quantlab.config.settings import BacktestConfig


MARKET_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class BacktestMetadata:
    """一次实际运行的身份、数据、配置和实现版本。"""

    schema_version: str
    run_id: str
    run_at: str
    strategy: dict
    data: dict
    backtest_config: dict
    analytics_config: dict | None
    execution_rule: str
    code_sha256: str
    environment: dict[str, str]


def snapshot_input(data: pd.DataFrame) -> pd.DataFrame:
    """只保存引擎实际使用的字段，固定类型与行序以支持跨文件复核。"""
    columns = MARKET_COLUMNS + (["signal"] if "signal" in data else [])
    snapshot = data.loc[:, columns].copy(deep=True)
    snapshot["date"] = pd.to_datetime(snapshot["date"])
    snapshot["symbol"] = snapshot["symbol"].astype("string")
    for column in MARKET_COLUMNS[2:]:
        snapshot[column] = snapshot[column].astype(float)
    if "signal" in snapshot:
        snapshot["signal"] = snapshot["signal"].astype("string")
    return snapshot.sort_values(["date", "symbol"]).reset_index(drop=True)


def dataframe_records(data: pd.DataFrame) -> list[dict]:
    """转换为标准 JSON 类型，保留浮点精度，缺失值表示为 null。"""
    clean = data.astype(object).where(pd.notna(data), None)
    records = clean.to_dict(orient="records")
    for row in records:
        for key, value in row.items():
            if isinstance(value, (datetime, pd.Timestamp)):
                row[key] = value.isoformat()
    return records


def dataframe_sha256(data: pd.DataFrame) -> str:
    payload = json.dumps(
        dataframe_records(data), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def source_tree_sha256(source_root: Path) -> str:
    """对实际 Python 源码取指纹，未提交的改动也会影响版本。"""
    digest = sha256()
    paths = sorted(source_root.rglob("*.py"))
    if not paths:
        raise ValueError("找不到用于记录版本的 Python 源码")
    for path in paths:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


# Python 进程可能仍持有旧函数，因此不能把刚改过的磁盘源码当成已执行版本。
# 仅重载本模块不能重置基线：引擎等其他模块可能依然没有重载。
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if "_LOADED_CODE_SHA256" not in globals():
    _LOADED_CODE_SHA256 = source_tree_sha256(_SOURCE_ROOT)


def create_metadata(
    input_data: pd.DataFrame,
    config: BacktestConfig,
    *,
    strategy_name: str,
    strategy_parameters: dict | None = None,
    execution_rule: str = "next_day_open",
    data_source: str | None = None,
    data_version: str | None = None,
) -> BacktestMetadata:
    if source_tree_sha256(_SOURCE_ROOT) != _LOADED_CODE_SHA256:
        raise ValueError("源码已更改，请重新启动 QuantLab 或 Python 进程后再运行回测")
    market_hash = dataframe_sha256(input_data[MARKET_COLUMNS])
    return BacktestMetadata(
        schema_version="backtest-run-v1",
        run_id=str(uuid4()),
        run_at=datetime.now(timezone.utc).isoformat(),
        strategy={"name": strategy_name, "version": "1", "parameters": dict(strategy_parameters or {})},
        data={
            "source": data_source,
            "version": data_version or f"sha256:{market_hash}",
            "market_sha256": market_hash,
            "input_sha256": dataframe_sha256(input_data),
            "row_count": len(input_data),
            "symbols": sorted(input_data["symbol"].unique().tolist()),
            "start_date": input_data["date"].iloc[0].isoformat(),
            "end_date": input_data["date"].iloc[-1].isoformat(),
            "price_column": "close",
        },
        backtest_config=asdict(config),
        analytics_config=None,
        execution_rule=execution_rule,
        code_sha256=_LOADED_CODE_SHA256,
        environment={
            "python": platform.python_version(),
            **{name: version(name) for name in ("pandas", "numpy", "pyarrow")},
        },
    )
