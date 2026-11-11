# 回测运行记录

这项收尾让一次回测的结果可以追溯到它实际使用的输入和设置。
指纹是内容的 SHA-256 摘要：内容发生改变时，指纹也会改变。

## 数据和代码分别记录什么

- `metadata.run_id`：每次实际运行的唯一 ID。
- `metadata.run_at`：UTC 运行时间，包含时区。
- `metadata.strategy`：策略名称、规则版本和参数。
- `metadata.data`：来源文件、导入版本、行情指纹、行情及信号的联合指纹、
  代码列表、区间、行数和价格列。
- `metadata.backtest_config`：初始资金、手续费、滑点、税率和每手数量。
- `metadata.analytics_config`：年化期数和无风险利率；仅运行原始引擎时为
  `null`，表示尚未计算绩效。
- `metadata.execution_rule`：次日开盘成交或买入持有的首日开盘建仓。
- `metadata.code_sha256`：进程首次加载回测记录模块时，`src/quantlab`
  全部 Python 源码的指纹；每次运行前检查磁盘源码未变化。
- `metadata.environment`：Python、pandas、NumPy、PyArrow 的版本。
- `input_data`、`orders`、`trades`、`equity_curve`、`metrics`：实际输入
  快照、订单、成交、资金曲线和绩效摘要。

没有来源文件的内存数据使用内容指纹作为版本。代码指纹包含未提交的
源码更改，不能把当前 Git HEAD 当作它的替代品。记录不扫描环境变量，
不包含 iFinD 凭证。

如果运行期间修改了源码，后续回测会提示重新启动 QuantLab 或 Python
进程。这是为了避免“内存中执行旧代码、记录却标成新代码”。只刷新网页
或单独重载记录模块不能解除检查；需要重启应用进程。

## 页面操作

在“策略实验”点击运行后，记录自动写入 `data/backtests/<run_id>.json`。
“方法对比”会保存两个模拟回测，真实交易上传不会写入这些记录。
相应结果的“实验记录与下载”展开项提供 JSON 下载。下载不会重新运行回测。

相同数据和参数再次点击运行会生成新的 ID 和时间，旧记录不会覆盖。
公开临时会话模式使用当前会话自己的临时目录，不写入本地持久记录目录。
本地磁盘写入失败时页面会提示，并保留下载入口。

## Python 调用

```python
from quantlab.backtest import run_dual_moving_average_backtest, save_backtest_run

result = run_dual_moving_average_backtest(
    market_data, 20, 60,
    data_source="data/processed/market/510300/example.parquet",
    data_version="这里填实际导入版本",
)
path = save_backtest_run(
    result.backtest, "data/backtests", performance=result.performance,
)
```

省略 `data_source` 和 `data_version` 时，仍会记录输入内容指纹。
数值计算函数本身不写磁盘，便于测试与后续阶段复用。

使用 `analyze_backtest_result(raw_result)` 后，应一起传入返回值的
`analysis.backtest` 和 `analysis.performance`，不要传入分析前的原始
`raw_result`。导出会检查指标配置，并依据资金曲线重算绩效摘要，拒绝
混用其他分析的指标。

下面示例从双均线记录中的实际输入重放交易执行，不依赖原行情文件：

```python
import json
from pathlib import Path
import pandas as pd
from quantlab.backtest import analyze_backtest
from quantlab.config.settings import AnalyticsConfig, BacktestConfig

record = json.loads(Path("data/backtests/实际运行ID.json").read_text())
metadata = record["metadata"]
replayed = analyze_backtest(
    pd.DataFrame(record["input_data"]),
    backtest_config=BacktestConfig(**metadata["backtest_config"]),
    analytics_config=AnalyticsConfig(**metadata["analytics_config"]),
)
```

买入持有记录不含信号列，重放时使用 `run_buy_and_hold`；想重新检验双均线
信号生成时，用保存的行情和 `strategy.parameters` 调用
`run_dual_moving_average_backtest`。原始信号回测若没有计算绩效，可以直接
使用 `run_backtest`。

精确复验应使用记录对应的源码和依赖版本。JSON 保存输入与结果、并标识
代码版本，但不打包旧版源码或整个 Python 环境。完整实验检索与版本管理
仍属于第三阶段。

## 实现位置

- `backtest/provenance.py`：规范化输入快照并记录数据、代码与环境版本。
- `backtest/archive.py`：生成 JSON 并原子保存；拒绝覆盖已有运行记录。
- `ui/run_records.py`：连接自动保存、保存失败提示与页面下载。
- `tests/test_backtest_provenance.py`、`test_backtest_archive.py`：验证指纹、
  源码变更后要求重启、不覆盖历史、输入快照不可冒用、指标配置一致，
  以及保存后重放成交和资金曲线。
- `tests/test_ui_run_records.py`：验证实际页面提交后的保存与导出行为。
