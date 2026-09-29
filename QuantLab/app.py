import importlib
from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _refresh_legacy_data_module_cache() -> None:
    """Refresh a pre-refactor data package during Streamlit hot reloads."""
    import quantlab.data as data_package

    if hasattr(data_package, "MarketDataService"):
        return

    for module_name in tuple(sys.modules):
        if module_name == "quantlab.data.market_service" or module_name.startswith(
            "quantlab.data.providers"
        ):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    importlib.reload(data_package)


_refresh_legacy_data_module_cache()

from quantlab.ui import require_access  # noqa: E402


st.set_page_config(
    page_title="QuantLab",
    page_icon=":material/monitoring:",
    layout="wide",
)

require_access()

st.markdown(
    """
    <style>
    :root {
        --ql-blue: #4b8df8;
        --ql-coral: #ff6b63;
    }
    .stMainBlockContainer {
        max-width: 1180px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stMetricValue"] {
        letter-spacing: -0.02em;
    }
    .ql-eyebrow {
        color: #7faeff;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .ql-muted {
        color: rgba(250, 250, 250, 0.58);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

navigation = st.navigation(
    {
        "我的投资": [
            st.Page(
                PROJECT_ROOT / "pages" / "0_首页.py",
                title="成长总览",
                icon=":material/space_dashboard:",
                default=True,
            ),
            st.Page(
                PROJECT_ROOT / "pages" / "1_数据管理.py",
                title="添加资产数据",
                icon=":material/add_chart:",
            ),
            st.Page(
                PROJECT_ROOT / "pages" / "6_多ETF数据.py",
                title="多 ETF 数据",
                icon=":material/table_chart:",
            ),
        ],
        "分析": [
            st.Page(
                PROJECT_ROOT / "pages" / "2_投资表现分析.py",
                title="投资表现",
                icon=":material/analytics:",
            ),
        ],
        "实验": [
            st.Page(
                PROJECT_ROOT / "pages" / "7_因子研究.py",
                title="因子研究",
                icon=":material/query_stats:",
                url_path="factor_research",
            ),
            st.Page(
                PROJECT_ROOT / "pages" / "3_双均线回测.py",
                title="策略实验",
                icon=":material/science:",
            ),
            st.Page(
                PROJECT_ROOT / "pages" / "4_三方对比.py",
                title="方法对比",
                icon=":material/compare_arrows:",
            ),
        ],
        "改进": [
            st.Page(
                PROJECT_ROOT / "pages" / "5_交易复盘.py",
                title="交易复盘",
                icon=":material/fact_check:",
            ),
        ],
    }
)
navigation.run()
