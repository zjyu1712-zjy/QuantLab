"""只读实验包：原始 JSON 备份和从同一快照导出的 CSV，不执行任何研究计算。"""

import csv
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from io import BytesIO, StringIO
import json
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

from quantlab.portfolio.archive import LoadedPortfolioRun, _decode_archive_bytes
from quantlab.portfolio.archive_codec import METRICS, TABLES, require


@dataclass(frozen=True)
class PortfolioExport:
    file_name: str
    content: bytes


def _csv_value(value):
    """CSV 为阅读副本：空值留空，数字不舍入，危险文本加前置单引号。"""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, str):
        if value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
    if isinstance(value, (datetime, date)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    # csv.writer 的数字文本转换不使用界面的两位小数/百分比格式。
    return value


def _csv_bytes(frame):
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(frame.columns)
    for row in frame.itertuples(index=False, name=None):
        writer.writerow([_csv_value(value) for value in row])
    return stream.getvalue().encode("utf-8-sig")


_FIELD_NOTES = """字段说明（英文列名保持稳定）：
date 日期；symbol 六位 ETF 代码；signal_date 信号日期；execution_date 执行日期。
decision_id 调仓编号；order_id 订单编号；trade_id 成交编号；status 状态；reason 原因。
valid_count 有效因子数量；top_n 选取数量；cutoff_tie 选取边界是否同值。
factor_value 原始因子值；target_weight 等权目标权重；is_valid 因子是否有效。
has_observation 是否有行情；open/high/low/close 开/高/低/收盘价（元）；volume 成交量（份）。
side BUY 买入 / SELL 卖出；requested_quantity/filled_quantity/unfilled_quantity 申请/成交/未成交份额。
quantity 持仓或成交份额；position_after 成交后持仓份额；cash_after 成交后现金（元）。
open_price 开盘价；execution_price 模拟成交价（含滑点），单位元。
notional 成交额；commission 佣金；tax 税费；slippage_cost 滑点影响，单位元。
cash 现金；position_value 持仓市值；equity 总资产，单位元。
weight 实际持仓权重；net_value 净值（起点 1）；cumulative_return 累计收益率。
cumulative_fees_taxes 累计佣金及税费；cumulative_slippage 累计滑点影响，单位元。
daily_return 单期简单收益率；drawdown 回撤（下跌为负数）。
metric 指标名；value 保存值；unit 单位；unavailable_reason 未定义原因。
annualized_return 年化收益率；annualized_volatility 年化波动率；max_drawdown 最大回撤；sharpe_ratio 夏普比率。
"""


def _readme(status):
    return ("QuantLab 单次实验导出包（portfolio-export-v1）\n\n"
            "archive.json 是原档案的逐字节备份，保留原运行身份、时间、版本、参数、指纹及完整数据快照。\n"
            "CSV 是便于复盘的阅读副本，不是无损恢复格式；当前未提供重新导入界面。\n"
            "metadata.json 保存档案身份、原参数、基准状态、表类型及 CSV 约定；不添加当前机器路径。\n"
            "strategy/ 为组合策略；benchmark/ 仅在基准完整成功时存在。\n"
            f"本档案基准状态：{status}（complete 完整成功；not_run 未运行；unavailable 运行失败）。\n\n"
            "每个目录内：\n"
            "metrics.csv 五项绩效指标和未定义原因；performance.csv 按完整日期轴列出保存的收益、净值、回撤。\n"
            "equity_curve.csv 资金账本（最后一行 equity 为期末资产）；positions.csv 每日持仓。\n"
            "decisions.csv 调仓决策；targets.csv 目标；orders.csv 委托；trades.csv 成交。\n"
            "input_data.csv 行情快照；factor_data.csv 因子快照（基准不含此表）。空表保留表头。\n\n"
            "所有指标来自保存值，未重新计算，不受当前研究参数影响。年化期数、无风险利率及价格口径见 metadata.json。\n"
            "收益率/波动率/回撤/权重使用比例数值，例如 0.05 表示 5%；夏普无量纲；不是已乘 100 的百分数。\n"
            "CSV 空单元格表示缺失、未定义或原空文本，绝不补零；细分类型和原值以 archive.json 为准。\n"
            "首日 daily_return 留空（没有前一期）；真实零值写为 0 或 0.0。日期使用 YYYY-MM-DD。\n"
            "数据未按界面显示位数舍入；Excel 的数值精度可能有限，精确复核请使用 JSON 和支持浮点往返的程序。\n\n"
            "Excel：使用“数据 → 从文本/CSV”导入，编码选择 UTF-8；symbol 和各类编号列设为文本，避免代码前导零丢失。\n"
            "以 =、+、-、@ 等开头的文本在 CSV 中前置单引号，防止公式执行；负数数值不加引号。\n"
            "此安全转换只影响 CSV 文本副本；JSON 中的原文本不变。请勿执行档案中保存的公式或任意文本。\n\n"
            "隐私：本包包含完整行情、因子快照和已保存元数据，自由文本可能含私人信息，分享前请检查。\n"
            "指纹用于发现损坏，不是数字签名、加密或投资收益证明。旧代码版本的结果不会因导出而重新认证。\n\n"
            + _FIELD_NOTES).encode("utf-8")


def build_portfolio_export(loaded: LoadedPortfolioRun) -> PortfolioExport:
    """从同次载入的原始字节构建内存 ZIP；不信任可变展示表，不读写磁盘。"""
    require(isinstance(loaded, LoadedPortfolioRun), "导出需要一份已加载的组合档案")
    # 重新核验字节本身；不会使用当前指标公式，也不重开原文件造成混配。
    snapshot = _decode_archive_bytes(loaded.archive_bytes, run_id=loaded.record.run_id, path=loaded.record.path)
    entries = {"archive.json": snapshot.archive_bytes, "README.txt": _readme(snapshot.benchmark_status)}
    types = {}
    for prefix, analysis in (("strategy", snapshot.strategy), ("benchmark", snapshot.benchmark)):
        if analysis is None:
            continue
        for name in TABLES:
            frame = getattr(analysis.backtest, name)
            if frame is not None:
                entries[f"{prefix}/{name}.csv"] = _csv_bytes(frame)
                types[f"{prefix}/{name}.csv"] = {key: str(dtype) for key, dtype in frame.dtypes.items()}
        performance = analysis.performance
        # 按已保存日期对齐序列，首日收益自然为空；不计算收益，不缩短日期轴。
        series = pd.DataFrame({"daily_return": performance.daily_returns,
                               "net_value": performance.net_value, "drawdown": performance.drawdown})
        entries[f"{prefix}/performance.csv"] = _csv_bytes(series.rename_axis("date").reset_index())
        metrics = pd.DataFrame([
            {"metric": key, "value": getattr(performance, key),
             "unit": "dimensionless" if key == "sharpe_ratio" else "ratio",
             "unavailable_reason": performance.unavailable_reasons.get(key, "")}
            for key in METRICS
        ], columns=["metric", "value", "unit", "unavailable_reason"])
        entries[f"{prefix}/metrics.csv"] = _csv_bytes(metrics)
    metadata = {
        "export_schema_version": "portfolio-export-v1", "run_id": snapshot.record.run_id,
        "name": snapshot.record.name, "saved_at": snapshot.record.saved_at,
        "archive_file_sha256": sha256(snapshot.archive_bytes).hexdigest(),
        "benchmark_status": snapshot.benchmark_status, "benchmark_failure": snapshot.benchmark_failure,
        "strategy": snapshot.strategy.backtest.metadata,
        "benchmark": snapshot.benchmark.backtest.metadata if snapshot.benchmark else None,
        "table_dtypes": types,
        "csv_conventions": {"encoding": "utf-8-sig", "missing": "", "ratios": "0.05 = 5%",
                            "dangerous_text_prefix": "'", "date": "YYYY-MM-DD"},
    }
    entries["metadata.json"] = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as bundle:
        for name, content in entries.items():
            # 固定路径和 ZIP 时间戳：同一快照反复导出得到相同内容，不混入导出时的机器信息。
            item = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = ZIP_DEFLATED
            bundle.writestr(item, content)
    return PortfolioExport(f"quantlab-portfolio-{snapshot.record.run_id}.zip", buffer.getvalue())
