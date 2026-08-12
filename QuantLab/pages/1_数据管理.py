from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlab.config.logging import configure_logging  # noqa: E402
from quantlab.data import (  # noqa: E402
    MarketDataService,
    RESERVED_MARKET_SOURCES,
    SUPPORTED_MARKET_SOURCES,
    get_market_data_provider,
    normalize_etf_symbol,
)
from quantlab.ui import (  # noqa: E402
    create_market_template_bytes,
    friendly_market_error,
    get_market_storage_dirs,
    import_uploaded_market_history,
    inspect_uploaded_market_history,
    is_public_mode,
    list_market_datasets,
    load_market_dataset,
    require_access,
)
from quantlab.ui.market_data import market_source_label  # noqa: E402


RAW_MARKET_DIR, PROCESSED_MARKET_DIR = get_market_storage_dirs(PROJECT_ROOT)
LOGGER = configure_logging(PROJECT_ROOT / "logs")


@st.cache_data(show_spinner=False)
def _load_cached_market_dataset(
    processed_path: str,
    modified_time_ns: int,
) -> pd.DataFrame:
    del modified_time_ns
    return load_market_dataset(processed_path)


def _format_dataset_option(index: int, records: list) -> str:
    record = records[index]
    row_text = f"{record.row_count} 条" if record.row_count is not None else "记录数未知"
    return f"{record.symbol} · {row_text} · 添加于 {record.imported_at[:10]}"


def _show_friendly_error(error: Exception) -> None:
    message, technical_detail = friendly_market_error(error)
    st.error(message)
    if technical_detail:
        with st.expander("查看详细原因"):
            st.code(technical_detail, language=None)


def _render_auto_fetch(symbol: str) -> None:
    st.markdown("### 自动获取行情")
    st.caption("输入 ETF 代码和日期，QuantLab 会自动获取、检查并保存历史行情。")
    date_columns = st.columns(2)
    default_end = date.today()
    try:
        default_start = default_end.replace(year=default_end.year - 5)
    except ValueError:  # 2 月 29 日
        default_start = default_end.replace(year=default_end.year - 5, day=28)
    start_date = date_columns[0].date_input(
        "开始日期",
        value=default_start,
        key="auto_market_start_date",
    )
    end_date = date_columns[1].date_input(
        "结束日期",
        value=default_end,
        key="auto_market_end_date",
    )
    with st.expander("高级设置"):
        adjustment = st.selectbox(
            "复权方式",
            options=["none", "qfq", "hfq"],
            format_func=lambda value: {
                "none": "不复权",
                "qfq": "前复权",
                "hfq": "后复权",
            }[value],
            help="不同复权方式会产生不同价格序列；同一份分析中请保持一致。",
            key="auto_market_adjustment",
        )
    if st.button(
        "获取历史行情",
        type="primary",
        disabled=not symbol,
        key="fetch_market_history",
    ):
        try:
            with st.status("正在准备历史行情……", expanded=True) as status:
                st.write("正在检查 ETF 代码")
                normalized_symbol = normalize_etf_symbol(symbol)
                st.write("正在检查本地数据并获取缺少的行情")
                service = MarketDataService(
                    get_market_data_provider(),
                    RAW_MARKET_DIR,
                    PROCESSED_MARKET_DIR,
                    logger=LOGGER.getChild("market_data"),
                )
                result = service.fetch_etf_history(
                    normalized_symbol,
                    start_date,
                    end_date,
                    period="daily",
                    adjustment=adjustment,
                )
                st.write("正在检查数据质量")
                st.write("正在保存数据")
                status.update(label="行情数据准备完成", state="complete")

            st.cache_data.clear()
            source_text = market_source_label(
                result.metadata.provider,
                result.cache_status,
            )
            st.success(
                "行情数据已保存，下次分析将优先使用本地数据。"
                if result.cache_status != "hit"
                else "已直接使用完整的本地行情数据。"
            )
            metrics = st.columns(5)
            metrics[0].metric("ETF 代码", result.symbol)
            metrics[1].metric("数据来源", source_text)
            metrics[2].metric(
                "行情区间",
                f"{result.data['date'].min().date()} 至 {result.data['date'].max().date()}",
            )
            metrics[3].metric("交易日数量", result.row_count)
            metrics[4].metric(
                "最近更新",
                result.metadata.fetched_at[:10],
            )

            preview = result.data.tail(8).copy()
            preview["date"] = preview["date"].dt.date
            st.markdown("#### 最近行情预览")
            st.dataframe(
                preview,
                hide_index=True,
                width="stretch",
                column_config={
                    "date": "日期",
                    "symbol": "ETF 代码",
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "volume": "成交量",
                },
            )
            st.page_link(
                "pages/2_投资表现分析.py",
                label="开始分析",
                icon=":material/arrow_forward:",
            )
            with st.expander("高级信息：本次行情获取"):
                st.write(f"数据来源：{result.metadata.provider}")
                st.write(f"周期：{result.metadata.period}")
                st.write(f"复权方式：{result.metadata.adjustment}")
                st.write(f"数据版本：{result.metadata.data_version}")
                st.write(f"缓存状态：{result.cache_status}")
        except Exception as error:
            LOGGER.getChild("market_data").exception("自动获取行情失败")
            _show_friendly_error(error)


def _render_upload(symbol: str) -> None:
    st.markdown("### 上传历史行情文件")
    st.caption(
        "支持常见行情来源。上传后系统会自动识别字段，并转换为 QuantLab 统一格式。"
    )
    source_keys = list(SUPPORTED_MARKET_SOURCES)
    source = st.selectbox(
        "文件来自哪里（可选）",
        options=source_keys,
        format_func=lambda key: SUPPORTED_MARKET_SOURCES[key],
        help="不确定时保持“自动识别”即可。",
    )
    uploaded_file = st.file_uploader(
        "选择历史行情文件",
        type=["csv"],
        help="当前支持 CSV 文件；系统会自动识别中英文字段和常见日期格式。",
    )

    inspection = None
    if uploaded_file is not None:
        try:
            inspection = inspect_uploaded_market_history(
                uploaded_file.name,
                uploaded_file.getvalue(),
                symbol=symbol or None,
                source=source,
            )
            st.success(
                f"已识别为 {inspection.source_label}："
                f"{inspection.symbol}，共 {inspection.row_count} 条行情。"
            )
            preview = inspection.standardized_data.head(8).copy()
            preview["date"] = preview["date"].dt.date
            st.dataframe(
                preview,
                hide_index=True,
                width="stretch",
                column_config={
                    "date": "日期",
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "volume": "成交量",
                },
            )
            with st.expander("高级信息：识别结果与字段转换"):
                st.write(f"识别来源：{inspection.source_label}")
                st.write(f"文件编码：{inspection.encoding}")
                st.json(inspection.field_mapping)
        except Exception as error:
            _show_friendly_error(error)

    if st.button(
        "添加到我的投资",
        type="primary",
        disabled=inspection is None,
        key="import_market_history",
    ):
        try:
            with st.spinner("正在自动检查并添加资产数据……"):
                result = import_uploaded_market_history(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    RAW_MARKET_DIR,
                    PROCESSED_MARKET_DIR,
                    symbol=symbol or None,
                    source=source,
                )
            st.cache_data.clear()
            st.success(
                f"已添加 ETF {result.symbol}，共 {result.row_count} 条行情。"
                "现在可以开始分析。"
            )
            st.page_link(
                "pages/2_投资表现分析.py",
                label="查看投资表现",
                icon=":material/arrow_forward:",
            )
        except Exception as error:
            _show_friendly_error(error)


def _render_template_download() -> None:
    st.markdown("### 下载 QuantLab 行情模板")
    st.caption(
        "如果系统暂时无法识别你的文件，可以把行情整理到这份空白模板中。"
    )
    st.download_button(
        "下载行情模板",
        data=create_market_template_bytes(),
        file_name="quantlab_market_template.csv",
        mime="text/csv",
        icon=":material/download:",
        type="primary",
    )
    st.code("date,open,high,low,close,volume", language="text")


def _render_supported_formats() -> None:
    with st.expander("查看支持格式"):
        st.markdown(
            """
**当前支持**

- QuantLab 标准行情文件
- 东方财富历史行情
- 同花顺历史行情
- 通达信历史行情
- Yahoo Finance

系统可以识别常见的中英文字段，例如“日期 / 交易日期 / date”、
“开盘价 / open”和“成交量 / vol / volume”。
            """
        )
        st.write("后续预留：" + "、".join(RESERVED_MARKET_SOURCES) + "。")
        st.caption(
            "iFinD 需要在服务端配置已授权的访问凭证，配置完成后即可使用。"
        )
        st.caption("Wind 和 Choice 当前暂未接入。")


def _render_saved_assets() -> None:
    st.divider()
    st.header("我的资产数据")
    try:
        records = list_market_datasets(PROCESSED_MARKET_DIR, RAW_MARKET_DIR)
    except Exception as error:
        _show_friendly_error(error)
        return

    if not records:
        st.info(
            "还没有资产数据。完成上方任一导入方式后，就可以开始分析和实验。",
            icon=":material/info:",
        )
        return

    selected_index = st.selectbox(
        "选择资产",
        options=range(len(records)),
        format_func=lambda index: _format_dataset_option(index, records),
    )
    selected = records[selected_index]
    selected_path = Path(selected.processed_path)

    try:
        market_data = _load_cached_market_dataset(
            str(selected_path),
            selected_path.stat().st_mtime_ns,
        )
    except Exception as error:
        _show_friendly_error(error)
        return

    first_date = pd.to_datetime(market_data["date"]).min().date()
    last_date = pd.to_datetime(market_data["date"]).max().date()
    metrics = st.columns(3)
    metrics[0].metric("ETF 代码", selected.symbol)
    metrics[1].metric("行情记录", len(market_data))
    metrics[2].metric("数据区间", f"{first_date} 至 {last_date}")

    st.markdown("#### 收盘价预览")
    st.line_chart(
        market_data.set_index("date")[["close"]],
        x_label="日期",
        y_label="收盘价",
        width="stretch",
    )

    with st.expander("高级信息：数据版本与标准化结果"):
        st.write(f"添加时间（UTC）：{selected.imported_at[:19]}")
        st.write(f"数据版本：{selected.version_id}")
        source_meta_path = (
            Path(RAW_MARKET_DIR) / selected.version_id / "source_format.json"
        )
        if source_meta_path.is_file():
            source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
            st.write(f"识别来源：{source_meta.get('source_label', '未知')}")
            st.write("原始文件已保留，计算使用统一标准化数据。")
        st.dataframe(
            market_data.head(100),
            hide_index=True,
            width="stretch",
        )


def render_data_management_page() -> None:
    st.markdown('<div class="ql-eyebrow">我的投资 · 第一步</div>', unsafe_allow_html=True)
    st.title("添加资产数据")
    st.caption("导入 ETF 历史行情，用于收益分析、策略实验和投资复盘。")

    if is_public_mode():
        st.info(
            "上传内容只用于当前浏览器会话，不会与其他访客共享；"
            "QuantLab 不连接券商，也不会自动下单。",
            icon=":material/lock:",
        )
    else:
        st.info(
            "所有文件只保存在你的电脑；QuantLab 不连接券商，也不会自动下单。",
            icon=":material/lock:",
        )

    symbol = st.text_input(
        "ETF 代码",
        placeholder="例如：159516",
        help="如果文件中已经包含唯一的资产代码，也可以留空让系统识别。",
    ).strip()

    auto_tab, upload_tab, template_tab = st.tabs(
        ["自动获取行情（推荐）", "上传历史行情文件", "下载模板"]
    )
    with auto_tab:
        _render_auto_fetch(symbol)
    with upload_tab:
        _render_upload(symbol)
    with template_tab:
        _render_template_download()

    _render_supported_formats()
    _render_saved_assets()


require_access()
render_data_management_page()
