"""将回测输入、运行记录和输出保存为可核对的本地 JSON。"""

from dataclasses import asdict
import json
from math import isclose
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from quantlab.analytics.performance import PerformanceAnalysis, analyze_performance
from quantlab.backtest.engine import BacktestResult
from quantlab.backtest.provenance import dataframe_records, dataframe_sha256


def serialize_backtest_run(
    result: BacktestResult,
    *,
    performance: PerformanceAnalysis | None = None,
) -> str:
    """导出一次运行；拒绝将已被修改的输入快照冒充为原始输入。"""
    if result.metadata is None or result.input_data is None:
        raise ValueError("该回测缺少运行记录，请使用当前版本重新运行")
    if dataframe_sha256(result.input_data) != result.metadata.data["input_sha256"]:
        raise ValueError("回测输入快照的指纹不一致，请重新运行")
    metrics = None
    if performance is not None:
        if result.metadata.analytics_config is None:
            raise ValueError("绩效指标缺少计算配置，请传入 analysis.backtest 与 analysis.performance")
        verified = analyze_performance(
            result.equity_curve.set_index("date")["equity"],
            **result.metadata.analytics_config,
        )
        names = (
            "cumulative_return", "annualized_return", "annualized_volatility",
            "max_drawdown", "sharpe_ratio",
        )
        if any(
            not isclose(
                float(getattr(performance, name)), float(getattr(verified, name)),
                rel_tol=1e-12, abs_tol=1e-12,
            )
            for name in names
        ):
            raise ValueError("绩效指标与本次资金曲线或记录的计算配置不一致")
        metrics = {name: float(getattr(verified, name)) for name in names}
    return json.dumps(
        {
            "metadata": asdict(result.metadata),
            "input_data": dataframe_records(result.input_data),
            "orders": dataframe_records(result.orders),
            "trades": dataframe_records(result.trades),
            "equity_curve": dataframe_records(result.equity_curve),
            "metrics": metrics,
        },
        ensure_ascii=False, indent=2, allow_nan=False,
    )


def save_backtest_run(
    result: BacktestResult,
    output_dir: str | Path,
    *,
    performance: PerformanceAnalysis | None = None,
) -> Path:
    """完整写入后再发布文件；运行 ID 重复时拒绝覆盖历史记录。"""
    serialized = serialize_backtest_run(result, performance=performance)
    run_id = result.metadata.run_id
    if str(UUID(run_id)) != run_id:
        raise ValueError("回测运行 ID 无效")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{run_id}.json"
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".run-", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
        # 同目录硬链接原子发布，已有目的文件会抛出 FileExistsError。
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
