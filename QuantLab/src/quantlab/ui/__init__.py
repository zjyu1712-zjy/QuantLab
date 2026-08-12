"""QuantLab 界面层使用的轻量服务。"""

from quantlab.ui.backtest import (
    MovingAverageBacktestView,
    build_moving_average_backtest_view,
)
from quantlab.ui.comparison import (
    APPROACH_LABELS,
    ThreeWayComparisonView,
    build_three_way_comparison_view,
)
from quantlab.ui.market_data import (
    FriendlyMarketImportResult,
    MarketDatasetRecord,
    create_market_template_bytes,
    friendly_market_error,
    import_uploaded_market_history,
    import_uploaded_market_csv,
    inspect_uploaded_market_history,
    list_market_datasets,
    load_market_dataset,
)
from quantlab.ui.performance import (
    MarketPerformanceView,
    build_market_performance_view,
)
from quantlab.ui.runtime import (
    get_market_storage_dirs,
    is_public_mode,
    require_access,
)
from quantlab.ui.trade_data import load_uploaded_trade_csv
from quantlab.ui.trade_review import (
    TIMING_LABELS,
    TradeReviewView,
    build_trade_review_view,
)

__all__ = [
    "APPROACH_LABELS",
    "TIMING_LABELS",
    "MovingAverageBacktestView",
    "MarketPerformanceView",
    "MarketDatasetRecord",
    "FriendlyMarketImportResult",
    "ThreeWayComparisonView",
    "TradeReviewView",
    "build_moving_average_backtest_view",
    "build_market_performance_view",
    "build_three_way_comparison_view",
    "build_trade_review_view",
    "get_market_storage_dirs",
    "create_market_template_bytes",
    "friendly_market_error",
    "import_uploaded_market_history",
    "import_uploaded_market_csv",
    "inspect_uploaded_market_history",
    "is_public_mode",
    "list_market_datasets",
    "load_market_dataset",
    "load_uploaded_trade_csv",
    "require_access",
]
