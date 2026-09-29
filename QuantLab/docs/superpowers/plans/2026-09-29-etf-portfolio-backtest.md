# Multi-ETF Portfolio Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定研究快照上实现可逐笔核对的单因子 Top N 等权多 ETF 回测核心。

**Architecture:** 新增独立 `quantlab.portfolio`，依次连接既有研究快照、因子、
数值筛选、现金/持仓引擎和指标函数。第一阶段代码不反向依赖组合模块。
将纯费用和买入预算从引擎拆到 `execution.py`，不改变已批准的模块责任边界。

**Tech Stack:** 项目现有 Python 3.12、pandas、NumPy、pytest；不增加依赖。

**Spec:** `docs/superpowers/specs/2026-09-29-etf-portfolio-backtest-design.md`
（2026-09-29 用户已批准；执行者必须完整阅读设计和本计划）。

**Status:** 用户已确认本计划并选择当前会话逐步实施；Task 1–5 已完成，Task 6 的端到端验收和回归已通过，等待整体独立审查。
Task 1 验收：相关测试 181 项通过；全项目 1014 项通过、2 项外部接口测试跳过；依赖无冲突。
Task 2 验收：费用与预算测试 110 项通过；全项目 1124 项通过、2 项外部接口测试跳过；依赖无冲突。
Task 3 验收：引擎测试 55 项通过；引擎、预算与筛选相关测试 289 项通过；全项目 1179 项通过、2 项外部接口测试跳过；依赖无冲突。
Task 4 验收：公开研究入口测试 45 项通过；连同因子及单 ETF 追溯共 88 项通过；全项目 1224 项通过、2 项外部接口测试跳过；依赖无冲突。
Task 5 验收：组合分析测试 67 项通过；连同原绩效/单 ETF 分析共 71 项通过；全项目 1291 项通过、2 项外部接口测试跳过；依赖无冲突。
Task 6 验收：新增端到端 52 项通过；全项目 1343 项通过、2 项外部接口测试跳过；依赖无冲突。独立审查待完成。

## Global Constraints

- 工作目录 `/Users/zhoujiayu/Documents/量化/QuantLab`；文件路径以下相对此目录。
- 沿用当前项目目录和分支，保护已有未提交工作；不创建外部项目，不自动推送。
- “默认每 20 期调仓，可修改间隔；期数按研究日期轴位置，不按自然日或月份。”
- “t 日收盘后生成配置，下一研究日期开盘模拟执行；先卖后买。”
- “有效因子不足 N 只：跳过本次调仓，保留原持仓和现金；未建仓则继续现金。”
- “调仓所需开盘价缺失：停止回测，指出日期和 ETF，不假定成交。”
- “持仓 ETF 收盘价缺失：停止回测，不补旧价格、不输出完整期间绩效。”
- L 是因子回看期数，信号轴位置固定 `L + k*K`，不因跳过而重新锚定。
- 初始现金 100,000；每手 100；佣金率 0.0003；最低佣金 0；卖出税率 0；滑点率 0.0001。
- 每年期数默认 252、无风险年利率默认 0；不改变第一阶段指标的报错契约。
- 原始因子值排序、代码升序破同值并列、恰好 N 个席位、目标权重未舍入 `1/N`。
- 日历/复权/静态资产池限制全部随结果保留；不承诺真实可成交性或策略有效性。
- 全现金是合法路径，未定义指标为 `None + 原因`；失败回测没有成功或截断结果。
- 不做 UI、组合基准、FIFO/胜率、自动存档/下载、联网、自动交易、新因子或依赖升级。
- 每步先观察测试失败，再实现、验证、解释；不得把历史测试数量当作本次验收。

## Review Focus

1. 有效研究表的重复/多级行索引标签不属于数据身份；不能误拒绝或修改调用方索引（Task 4）。
2. 零成交量但没有实际成交需求时不应失败；实际份额为零的拒单也不收佣金（Task 3）。
3. 缺失日期恰遇跳过调仓，仍必须估值原持仓；不能把“跳过”当成跳过整天（Task 3）。
4. 巨大但有限的金额/份额不可引起逐手递减的超长循环或浮点无穷传播（Task 2、3）。
5. 返回表可变不意味着共享输入；修改返回值不得影响输入，再分析也不得修改原运行身份（Task 4、5）。

---

## 开工条件、文件地图与依赖

以下准备并入 Task 1，不单独制造一个无业务验收的脚手架任务。
计划获批后先记录 `git status --short`、实际测试基线与依赖检查：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -rs
.venv/bin/python -m pip check
```

目前有既有未提交修改，且 `src/quantlab/backtest/provenance.py` 是工作区
已有未跟踪文件。本计划复用它，但不擅自提交它或其他旧改动。单独检出当前
提交未必包含全部工作区依赖；交付时须如实说明，不能声称得到干净独立版本。
基线失败先定位并报告，不为本任务顺手修改无关模块。

| 文件 | 创建/修改与责任 |
| --- | --- |
| `src/quantlab/portfolio/types.py` | 新建：配置、调仓安排、结果和错误类型；分任务增加已用类型 |
| `src/quantlab/portfolio/selection.py` | 新建：调仓日期、原始值选取与决策记录 |
| `src/quantlab/portfolio/execution.py` | 新建：纯费用/整手预算函数，无日期或因子依赖 |
| `src/quantlab/portfolio/engine.py` | 新建：逐日状态、订单、成交、持仓和资金曲线 |
| `src/quantlab/portfolio/research.py` | 新建：唯一公开运行入口和组合运行身份 |
| `src/quantlab/portfolio/analysis.py` | 新建：完整资金曲线的指标适配 |
| `src/quantlab/portfolio/__init__.py` | 新建：按已完成任务导出公开类型和入口 |
| `tests/conftest.py` | 只追加 `make_portfolio_preview` 合成夹具，不改原有夹具 |
| `tests/test_portfolio_{selection,execution,engine,research,analysis,acceptance}.py` | 分别新建六个测试文件 |
| `docs/portfolio_backtest.md` | 新建学习、调用和验收说明，随对应功能递增 |
| `docs/roadmap.md` | 最后仅更新组合核心范围；保留已有未提交文档改动 |

不修改旧 `backtest/engine.py`、行情适配器、现有配置类、指标算法、研究集合
协议、因子公式、旧页面和依赖清单。精确暂存本任务文件；共享文件只暂存
本任务新增的片段，不能 `git add .` 或纳入他人的改动。

依赖：Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6。
逐步实现，不并行编辑相互依赖的类型；独立只读检查可以并行。每个任务的
一轮红灯/绿灯只是最低要求，新增独立行为仍需各自先观察失败。

## Task 1：固定调仓时间表与数值目标

**学习目标：** 把“每天查看排名”与“固定日期产生调仓配置”区分开。

**Files:** Create `portfolio/types.py`、`portfolio/selection.py`、
`portfolio/__init__.py`（均在 `src/quantlab/` 下）、`tests/test_portfolio_selection.py`；
Modify `tests/conftest.py`；Create `docs/portfolio_backtest.md`。

**Interfaces:**

- `PortfolioConfig(top_n: int, rebalance_interval: int = 20, ascending: bool | None = None, backtest: BacktestConfig = DEFAULT_BACKTEST_CONFIG)`，冻结数据类；N 上限在知道资产数量时检查。
- `PortfolioBacktestError(ValueError)`：初始化 `message: str, *, date: date | None = None, symbols: tuple[str, ...] = (), reason: str`；保留字段，中文消息，不包含输入快照或半成品。
- `RebalanceSchedule(decisions: pd.DataFrame, targets: pd.DataFrame, ascending: bool)`。
- `build_rebalance_schedule(result: FactorRunResult, *, config: PortfolioConfig) -> RebalanceSchedule`；内部接口，不从包顶层导出。
- `make_portfolio_preview(*, periods=65, gaps=None, supplied=False, adjustment="none", bump_after=None, zero_volume=None) -> tuple[ResearchSetPreview, Path]` 为新增 pytest 工厂夹具。

夹具以 2025-01-02 起的 `periods` 个 B 频日期为合成轴（不是官方日历）；
两只代码 510300/159915，OHLC 分别全部为 `100 + 0.5*i` / `200 - 0.5*i`，
正常 volume=1000。`gaps`/`zero_volume` 是代码到轴位置列表；`bump_after`
从给定位置起将两只全部 OHLC 乘 2；测试调用最大 125 期保证价格为正。
使用 tmp_path 下独立子目录真实导入、扫描、构建，返回源目录供移走测试。
`supplied=True` 才解析完整日期 CSV。不要复用既有因子夹具的零成交量默认值。

内部表列固定：decisions=`decision_id,signal_date,execution_date,valid_count,
top_n,status,reason,cutoff_tie`；targets=`decision_id,signal_date,symbol,
factor_value,target_weight`。ID 按信号日顺序 `PD000001` 起。可执行决策在此
内部阶段为 `PENDING`，最终由引擎变为 `EXECUTED`；不足 N 为
`SKIPPED_INSUFFICIENT_FACTORS`，末日足 N 为 `EXPIRED`。过期目标仍保留
用于解释，但不可成交；无下一日的 execution_date 为 NaT。

- [x] **Step 0：核对工作区并运行上述基线检查。** 890 项通过、2 项外部接口测试跳过；依赖无冲突。
- [x] **Step 1：写失败测试。** 在新文件先写以下断言；夹具真实
  构建流程参考 `tests/test_factor_research.py::make_preview`，不导入其私有夹具。

```python
def test_schedule_is_anchored_after_warmup(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45)
    factor = run_research_factor(preview, "momentum_20")
    result = build_rebalance_schedule(factor, config=PortfolioConfig(top_n=1))
    dates = pd.to_datetime(factor.calendar.dates)
    assert result.decisions.signal_date.tolist() == [dates[20], dates[40]]
    assert result.decisions.execution_date.tolist() == [dates[21], dates[41]]
    assert result.targets.symbol.tolist() == ["510300", "510300"]
    assert result.targets.target_weight.tolist() == [1.0, 1.0]

def test_skip_does_not_move_anchor(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45, gaps={"159915": [5]})
    result = build_rebalance_schedule(run_research_factor(preview),
                                     config=PortfolioConfig(top_n=2))
    assert result.decisions.status.tolist() == ["SKIPPED_INSUFFICIENT_FACTORS", "PENDING"]
    assert len(result.targets) == 2
```

另参数化：L=20/60/120、K=1/20、短历史、末日足/不足 N；config 拒绝
0/负数/bool/小数 N/K/手数、NaN/inf 金额和费率、滑点 1、非法方向及 N
超资产数。对真实 FactorRunResult 的独立副本设置原始数值测试筛选，不能
把这种注入加入公开回测入口：同值、显示舍入相同但原值不同、有效 0/负值、
全无效和缺失都按设计；与 `top_n_factor_preview` 数值选择/权重逐日核对。

- [x] **Step 2：运行红灯。** 124 项均因新组合模块不存在而失败，夹具正常完成。
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_selection.py -q`
  Expected: 新组合模块/接口不存在导致失败，而不是夹具导入错误。
- [x] **Step 3：实现上述类型和 `build_rebalance_schedule`。** 配置不静默
  转型；日期/代码键完整唯一，有效状态与有限值一致；复制而不修改输入。
  用 `range(L, M, K)`、按原值及代码排序选取；不调用 UI 私有函数或读文件。
- [x] **Step 4：运行绿灯与排名回归。** 181 项通过。
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_selection.py tests/test_ui_factors.py -q`
  Expected: 全部通过；日历/排序/输入不变均有断言。
- [x] **Step 5：新增学习说明。** 解释热身、跳过、过期、初始现金阶段和方向含义。
- [x] **Step 6：核对仅本任务差异后提交。** 消息 `feat: add portfolio rebalance scheduling`；未纳入既有未提交改动。

## Task 2：费用与独立买入预算

**学习目标：** 理解目标等权为什么不等于扣费后精确等权，以及余款为何留现金。

**Files:** Create `src/quantlab/portfolio/execution.py`、`tests/test_portfolio_execution.py`；
Modify `docs/portfolio_backtest.md`。

**Interfaces:** Consumes Task 1 `PortfolioBacktestError` 和既有 `BacktestConfig`。
Produces 以下内部纯函数，不从包顶层导出：

- `commission(notional: float, config: BacktestConfig) -> float`：仅正成交额调用；零成交额返回 0。
- `execution_price(open_price: float, side: str, config: BacktestConfig) -> float`：side 仅 BUY/SELL。
- `affordable_lot_quantity(budget: float, price: float, max_quantity: int, config: BacktestConfig) -> int`。
- `allocate_buys(cash: float, requests: dict[str, int], opens: dict[str, float], config: BacktestConfig) -> dict[str, int]`：返回每个申请代码的实际份额，包括 0。

- [x] **Step 1：写失败测试。**

```python
def test_minimum_fees_reduce_both_buys_without_code_priority():
    config = BacktestConfig(commission_rate=0, minimum_commission=5, slippage_rate=0)
    actual = allocate_buys(100000.0, {"510300": 5000, "159915": 5000},
                           {"510300": 10.0, "159915": 10.0}, config)
    assert actual == {"510300": 4900, "159915": 4900}
    assert 100000 - sum(q * 10 + 5 for q in actual.values()) == 1990

def test_slippage_and_each_side_cost():
    config = BacktestConfig(commission_rate=0.01, minimum_commission=2,
                           slippage_rate=0.01, sell_tax_rate=0.005, lot_size=10)
    assert execution_price(10, "BUY", config) == pytest.approx(10.1)
    assert execution_price(12, "SELL", config) == pytest.approx(11.88)
    assert commission(909, config) == pytest.approx(9.09)
    assert affordable_lot_quantity(1000, 10.1, 100, config) == 90
```

另测零现金/空申请/申请 0 不收最低佣金；不足一手为 0；现金够时不缩减；
最低佣金与按比例佣金的交界；不同价格和申请额按含费需求分预算；字典
插入顺序不影响结果；不把余款重新分配；最大份额很大也不逐手递减。
非法/非有限参数和运算溢出为 `PortfolioBacktestError`，无 NaN/inf 结果。

- [x] **Step 2：运行红灯。** 107 项均因新执行模块缺失而报错；另补精度回归，观察非零滑点丢失的 2 项断言失败。
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_execution.py -q`
  Expected: 新纯函数缺失对应的失败。
- [x] **Step 3：实现四个纯函数。** 完整需求金额按代码升序用 `math.fsum`
  累加，先一次性决定预算，后求各自最大可买手数。使用整数手数二分或代数
  上界后验证，不能从巨大需求逐手递减。仅对 `d_i>0` 计需求及最低佣金。
  不四舍五入成交额到分；量和现金判断使用真实精度。`MONEY_ATOL=1e-9`
  仅用于运算后近零余额校验/归整，不用于买入多一手；不使用相对容差融资。
- [x] **Step 4：运行绿灯。** 110 项通过。
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_execution.py -q`
  Expected: 全部通过，包括预算合计、顺序独立与大数量用例。
- [x] **Step 5：更新预算示例。** 文档用 100000 元及两笔最低佣金解释 1990 元余款。
- [x] **Step 6：核对差异后提交本任务文件。** 消息 `feat: add cost-aware portfolio allocation`；仅费用/预算模块、测试和相关文档，未纳入既有改动。

## Task 3：多资产现金、订单与每日估值

**学习目标：** 每一笔交易如何改变现金/份额，如何得到资金曲线。

**Files:** Create `src/quantlab/portfolio/engine.py`、`tests/test_portfolio_engine.py`；
Modify `src/quantlab/portfolio/types.py`、`docs/portfolio_backtest.md`；
`selection.py` 仅改为导入集中到 types.py 的决策/目标列定义，不改筛选行为。

**Interfaces:**

- Consumes `RebalanceSchedule`、`PortfolioConfig`、Task 2 四个预算函数及
  `data.alignment.validate_aligned_data(data, axis, symbols)`。
- Produces 内部 `simulate_portfolio(data: pd.DataFrame, calendar: CalendarAxis, schedule: RebalanceSchedule, *, config: PortfolioConfig) -> PortfolioBacktestResult`。
- `PortfolioBacktestResult` 冻结数据类字段：`decisions, targets, orders,
  trades, positions, equity_curve: pd.DataFrame`，以及 `metadata: dict | None = None,
  input_data: pd.DataFrame | None = None, factor_data: pd.DataFrame | None = None`。
  内部引擎尚未附运行身份，Task 4 的公开入口必须补齐，不能公开半成品。

在 types.py 固定列常量和空表类型：日期为无时区 datetime64[ns]；代码和
ID/状态为字符串；数量 int64；金额/权重 float64；布尔列不含空值。
超出可表示数量范围时失败，不溢出到负数。持仓/目标按日期、代码排序；
订单/成交按日期、先卖后买、同方向内代码排序，保留实际现金变动顺序。

- orders：`order_id,decision_id,symbol,signal_date,execution_date,side,
  requested_quantity,filled_quantity,unfilled_quantity,status,reason,trade_id`。
- trades：`trade_id,order_id,decision_id,symbol,signal_date,execution_date,
  side,quantity,open_price,execution_price,notional,commission,tax,slippage_cost,
  cash_after,position_after`。
- positions：`date,symbol,quantity,close,position_value,weight`，全日期×全代码；
  空仓缺失 close 留 NaN，但 quantity/value/weight 为 0。
- equity_curve：`date,cash,position_value,equity,net_value,cumulative_return,
  cumulative_fees_taxes,cumulative_slippage`。
- 订单 ID `PO000001`、成交 ID `PT000001`；拒单 trade_id 空值。

- [x] **Step 1：写手算账本与失败测试。** 只在引擎单测构造内部安排：
  D0 收盘选 A/B，D1 开盘执行；D1 收盘选 B/C，D2 开盘执行；N=2，
  初始 1000，手数 10，费用税滑点全 0。A/B/C 对应 510300/159915/512000。
  D0 开盘/收盘 10/20/5；D1 开盘 10/20/5、收盘 11/18/5；D2 开盘和收盘
  12/20/5。逐行 high=max(open,close)、low=min(open,close)、volume=1000。

```python
def test_cash_and_holdings_match_manual_ledger(manual_case):
    data, axis, schedule, config = manual_case
    result = simulate_portfolio(data, axis, schedule, config=config)
    assert result.equity_curve.cash.tolist() == [1000.0, 100.0, 150.0]
    assert result.equity_curve.equity.tolist() == [1000.0, 1010.0, 1100.0]
    last = result.positions.loc[result.positions.date.eq(pd.Timestamp(axis.dates[2]))]
    assert dict(zip(last.symbol, last.quantity)) == {"510300": 0, "159915": 20, "512000": 110}
    day_two = result.trades.loc[result.trades.execution_date.eq(pd.Timestamp(axis.dates[2]))]
    assert day_two.side.tolist() == ["SELL", "BUY"]
    assert day_two.symbol.tolist() == ["510300", "512000"]
```

本文件定义 `manual_case` pytest 夹具：三日期完整规范行情，手动 schedule
的 D0/D1 为 PENDING，目标分别 A/B 和 B/C，各 0.5；使用真实
`align_market_data` 创建日历和网格。仅引擎单测允许手动安排；公开入口仍
通过真实因子热身。另定义本文件小型安排工厂用于以下命名用例：

- `test_costs_are_not_double_counted`：1000 买 90 份、价 10.1，现金 81.91；
  之后全部以 11.88 卖，现金 1135.072；累计滑点 19.8、佣金 19.782、税 5.346。
  使用 N=1、Task 2 相同费率、买入原价 10 和卖出原价 12；第二次目标换入价格
  高到不足一手的另一资产，禁止新的实际买入。
- `test_zero_volume_without_fill_is_allowed`：目标数量已等于持仓或预算买不到
  一手时，volume=0 不误触发；有实际买/卖数量时零成交量必须抛错。
- `test_skipped_rebalance_still_values_existing_positions`：先持有，再安排
  SKIPPED；当日原持仓整行缺失仍抛错误，非持仓整行缺失不抛错误。
- `test_required_open_is_checked_before_any_batch_trade`：计划卖 A/买 B，
  B 开盘所在观测缺失，失败而不是只卖出 A 后返回部分结果。
- `test_expired_signal_never_executes`、`test_sold_asset_later_gap_is_allowed`、
  `test_same_target_no_orders_or_fees`、`test_cash_only_missing_axis_day_is_valid`。
- `test_partial_ohlcv_row_rejected`、`test_extreme_sell_fee_fails`（卖出净收入
  负）、`test_overflow_fails_without_mutation`；核对中文错误、日期、代码与原因。

- [x] **Step 2：运行红灯。** 初始 52 项均因引擎模块缺失失败；夹具构造成功。
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_engine.py -q`
  Expected: 新引擎缺失；夹具应可经原对齐接口合法构造。
- [x] **Step 3：实现逐日引擎。** 规范副本索引后验证完整日期网格和内部安排，
  拒绝重复目标、非恰好 N 个等权目标、执行日非紧邻信号日及区间外日期。
  每天在状态副本中：执行前日目标→先卖→同时分配买入预算→买→校验
  实际成交资产 volume→收盘估值→现金/份额/有限性检查→接受该日。
  严格区分跳过与空目标，期末不强平；对零数量不乘缺失价格。
  校验最后日缺失和失败时输入不变，失败不返回部分 result。
- [x] **Step 4：运行绿灯和记账不变量。** 引擎 55 项、预算 110 项通过；连同筛选共 289 项通过。
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_engine.py tests/test_portfolio_execution.py -q`
  Expected: 全部通过；逐日 equity=cash+市值，权重和现金占比之和为 1，
  全部数量为非负整手，费用/滑点分开对账，非计划日不成交。
- [x] **Step 5：更新三日账本说明。** 展示现金、各资产份额与总资产的对应关系；明确总资产耗尽时停止报错，未定义零资产权重。
- [x] **Step 6：核对差异后提交。** 消息 `feat: simulate multi-ETF portfolio accounting`；仅引擎、集中类型定义、筛选导入、测试和相关文档，不纳入既有改动。

## Task 4：研究快照接入与运行身份

**学习目标：** 保证结果对应明确的输入版本、公式版本和运行规则。

**Files:** Create `src/quantlab/portfolio/research.py`、`tests/test_portfolio_research.py`；
Modify `src/quantlab/portfolio/__init__.py`、`docs/portfolio_backtest.md`。

**Interfaces:** Consumes Task 1/3；公开导出：
`run_factor_portfolio(preview: ResearchSetPreview, *, factor_name: str,
config: PortfolioConfig) -> PortfolioBacktestResult`。
不允许 registry/external factor table 参数，不创建或写入数据目录。

metadata 为独立 dict：`schema_version="portfolio-run-v1"`、`status="complete"`、
`run_id,run_at,code_sha256,environment`、`strategy`（名称 single_factor_top_n、
version="1"、完整 config、resolved_ascending 和 L/K 锚点）、`factor_definition`、
`data`（预览/对齐/行情/因子内容指纹、白名单 sources、start/end、adjustment、
日期轴全部声明和 ISO 日期）、`warnings`；初始 `analytics_config=None`。
行情快照包含 has_observation；公式与输入身份各自保留，不伪造历史可得性。

- [x] **Step 1：写真实链路失败测试。** 覆盖五个内置因子、身份校验、输出隔离与独立子进程的源码变动保护。

```python
def test_real_factor_runs_only_at_next_open(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45)
    result = run_factor_portfolio(preview, factor_name="momentum_20",
                                  config=PortfolioConfig(top_n=1))
    dates = pd.to_datetime(preview.alignment.calendar.dates)
    assert result.trades.iloc[0].signal_date == dates[20]
    assert result.trades.iloc[0].execution_date == dates[21]
    assert result.equity_curve.iloc[:21].equity.eq(100000).all()
    assert result.metadata["schema_version"] == "portfolio-run-v1"
    assert result.metadata["status"] == "complete"
    assert result.metadata["data"]["preview_sha256"] == preview.preview_sha256
    assert not any("processed_path" in item for item in result.metadata["data"]["sources"])
```

补全：五个内置因子分别用至少 L+2 个日期；未知名拒绝；原始量/价格/来源/
日历/警告/指纹被改拒绝而非修复；重复、非连续、多级且名为 date/symbol
的行索引标签仍合法且不被修改。更改返回值不改变输入，再运行业务输出
相同（运行 ID/时间除外）。qfq/hfq 保留模型单位警告；不复权保留总回报限制。
全现金无交易仍返回完整身份和空表列类型。

另写 `test_portfolio_rejects_disk_code_changed_after_load`：仿照
`tests/test_backtest_provenance.py`，在 tmp_path 复制源码的子进程内运行，
改复制的 portfolio 源码并仅 reload provenance，第二次运行须提示重新启动。
只修改隔离副本，禁止改当前源码或修改全局环境来绕过校验。

- [x] **Step 2：运行红灯。** 43 项均因公开入口缺失失败，真实夹具构造成功。
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_research.py -q`
  Expected: 公开入口缺失；不是因夹具默认 volume=0 而失败。
- [x] **Step 3：实现唯一公开入口。** 深拷贝输入，用 `run_research_factor`
  内置工厂完成完整预览验证；对自己的行情副本清理行索引名并重置标签，
  不能重算身份以修复损坏输入。调用 schedule→simulate，最后装配独立结果。
  复用 `backtest.provenance.create_metadata` 仅取得运行 ID/时间、环境、
  受加载基线校验的源码身份；其余单 ETF schema/data/strategy 字段不照搬。
  行情和因子指纹用该模块的 `dataframe_sha256`；不访问私有基线变量。
  即使已选择的快照来源文件不存在也应运行；异常加上下文但不能吞掉失败。
- [x] **Step 4：运行绿灯。** 88 项通过；新增 2 项既有数值配置/冻结配置重验的兼容测试。
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_research.py tests/test_factor_research.py tests/test_backtest_provenance.py -q`
  Expected: 全部通过，旧追溯与因子契约未改变。
- [x] **Step 5：更新公开调用与追溯说明。** 区分内容指纹、版本声明与真实历史可得性。
- [x] **Step 6：核对差异后提交。** 消息 `feat: connect portfolio backtests to research snapshots`；仅本任务五个文件，不混入既有工作。

## Task 5：完整资金曲线的绩效分析

**学习目标：** 区分回测失败、指标未定义，以及真正的零收益。

**Files:** Create `src/quantlab/portfolio/analysis.py`、`tests/test_portfolio_analysis.py`；
Modify `types.py`、`__init__.py`（同一 portfolio 目录）、`docs/portfolio_backtest.md`。
另将 `research.py` 的一条阶段性说明改为通用年化假设，避免分析结果仍声称“尚未计算绩效”；不改研究行为。

**Interfaces:**

- `PortfolioPerformance`：`daily_returns,net_value,drawdown: pd.Series`；
  `cumulative_return,annualized_return,annualized_volatility,sharpe_ratio: float | None`；
  `max_drawdown: float`；`unavailable_reasons: dict[str, str]`。
- `PortfolioAnalysis(backtest: PortfolioBacktestResult, performance: PortfolioPerformance)`。
- 公开 `analyze_portfolio_result(result: PortfolioBacktestResult, *, analytics_config:
  AnalyticsConfig = DEFAULT_ANALYTICS_CONFIG) -> PortfolioAnalysis`。
- 输入必须是 Task 4 完整结果；返回独立 backtest 副本，metadata.analytics_config
  记录实际参数。原结果不得被分析过程修改，第一阶段 `analyze_performance` 不改。

- [x] **Step 1：写失败测试。** 使用 Task 4 短历史全现金结果：

```python
def test_cash_path_is_valid_but_sharpe_is_undefined(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=10)
    result = run_factor_portfolio(preview, factor_name="momentum_20",
                                  config=PortfolioConfig(top_n=2))
    actual = analyze_portfolio_result(result)
    assert actual.performance.cumulative_return == 0
    assert actual.performance.annualized_return == 0
    assert actual.performance.annualized_volatility == 0
    assert actual.performance.max_drawdown == 0
    assert actual.performance.sharpe_ratio is None
    assert "sharpe_ratio" in actual.performance.unavailable_reasons
    assert result.metadata["analytics_config"] is None
    assert actual.backtest.metadata["analytics_config"]["periods_per_year"] == 252
```

另测 1 个资产观测的净值 1/最大回撤 0、四项指标 None；2 个资产观测的
波动/夏普 None；3 个及以上正常变动路径逐项等于旧 analyze_performance。
首次成本和全部热身均被计入；返回分析对象改动不污染 result。
拒绝 None、缺 metadata/不完整日期轴、NaN/inf/非正资产、日期重复/乱序、
非法年化参数；日收益率标准差 <=1e-15 夏普未定义。合法极端数据若算得
非有限指标，应明确数值错误，不转换成“样本不足”。

- [x] **Step 2：运行红灯。** 63 项新行为因入口缺失失败，1 项原统一入口兼容测试通过。
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_analysis.py -q`
  Expected: 新分析入口缺失。
- [x] **Step 3：实现适配层和结果类型。** 按原始完整日期轴验证总资产，
  逐项调用 `analytics.returns` / `analytics.risk`，预先区分样本数量和
  夏普阈值；不复制公式，不用 broad except 填 None。单观测 daily_returns
  为空浮点序列；仍调用已有净值/回撤函数。记录不可计算原因并复制 metadata。
- [x] **Step 4：运行绿灯与旧指标回归。** 71 项通过；新增的 3 项损坏结果边界亦先红后绿。
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_analysis.py tests/test_performance.py tests/test_backtest_analysis.py -q`
  Expected: 全部通过，旧统一入口仍按原规则报未定义指标错误。
- [x] **Step 5：更新指标解释。** 给出全现金和单观测的未定义指标示例，以及首次费用和全部热身的手算对照。
- [x] **Step 6：核对差异后提交。** 消息 `feat: analyze complete portfolio equity curves`；仅本任务七个文件，不混入既有修改。

## Task 6：端到端、时序隔离与交付验收

**学习目标：** 证明各层连接后仍满足设计，而不只是各自单测正确。

**Files:** Create `tests/test_portfolio_acceptance.py`；Modify
`docs/portfolio_backtest.md`、`docs/roadmap.md`（仅本任务新增片段）。
若验收发现缺陷，仅修改对应 Task 的新组合文件并补红灯，不顺手改旧模块。

**Interfaces:** Consumes 两个公开入口和现有 `save_research_set(preview,
output_dir, *, name)`、`load_research_set(output_dir,set_id).preview`。
Produces 可重放的合成端到端证据、学习说明及准确范围状态，无新业务 API。

- [ ] **Step 1：写验收测试。** `test_saved_snapshot_runs_without_original_files`
  用 65 日期合成集合真实保存，将夹具源目录改名（不删除），真实重读、运行、
  分析；输入身份一致、研究文件字节不变、无自动回测档案或其他新增文件。
  `test_future_changes_do_not_change_earlier_execution` 分别用正常集合与
  bump_after=30 的有效新集合，比较 date/信号/执行日期 <D30 的决策、目标、
  订单、成交、持仓、资产逐项相同，不要求整个数据指纹相同。

  `test_supplied_axis_keeps_all_missing_cash_day` 在热身期让所有资产同一天
  缺失；supplied 轴保留空现金日期，union 轴不保留该日，两者不等同输入。
  `test_stage1_has_no_portfolio_or_factor_dependency` 检查旧数据/指标/策略/
  回测模块没有新增导入 portfolio/factors；因子原有单向依赖不误报。
  `test_public_roundtrip_accounting_invariants` 对不同 N、K、手数/费用、
  五个因子、正反方向运行并核对权重、非负余额、输入不变和费用一致。

```python
def test_saved_snapshot_runs_without_original_files(make_portfolio_preview, tmp_path):
    preview, source_root = make_portfolio_preview(periods=65)
    saved = save_research_set(preview, tmp_path / "sets", name="组合验收合成集合")
    source_root.rename(source_root.with_name(source_root.name + "_moved"))
    loaded = load_research_set(tmp_path / "sets", saved.set_id).preview
    result = run_factor_portfolio(loaded, factor_name="momentum_20",
                                  config=PortfolioConfig(top_n=2))
    analysis = analyze_portfolio_result(result)
    assert result.metadata["data"]["preview_sha256"] == preview.preview_sha256
    assert len(result.equity_curve) == 65
    assert len(result.positions) == 130
    assert result.trades.quantity.gt(0).all() and not result.trades.empty
    assert analysis.performance.cumulative_return is not None
```

- [ ] **Step 2：运行新增验收，逐项核对是否发现缺口。**
  Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_acceptance.py -q`
  Expected: 如失败，错误与真实需求缺口一致；已有实现满足的断言可直接通过，
  不为了“制造红灯”破坏代码。新发现缺陷另补最小失败测试再修复。
- [ ] **Step 3：完成交付说明。** 文档逐层解释调度/预算/账本/输入身份/绩效，
  给出真实公开入口示例、返回表解读、错误与模型限制。路线图仅标记组合
  核心已完成（确实通过后）；界面、基准和档案仍待实施，不把第二阶段勾完。
- [ ] **Step 4a：运行全项目回归。**
  Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -rs`
  Expected: 全部可运行测试通过，跳过有实际原因。
- [ ] **Step 4b：检查依赖。**
  Run: `.venv/bin/python -m pip check`
  Expected: 无依赖冲突。
- [ ] **Step 4c：检查差异。**
  Run: `git diff --check`
  Expected: 无空白错误，额外核对没有无关文件变更。
- [ ] **Step 4d：完成独立审查。** 只覆盖本任务变更及其调用契约。按所选执行方式落实审查，
  记录实际数字；未运行浏览器、外部行情测试不得宣称它们通过。
- [ ] **Step 5：记录实际验收结果。** 写入说明文档，不重复使用历史计数。
- [ ] **Step 6：提交任务片段。** 建议消息 `test: validate portfolio research workflow`。
- [ ] **Step 7：交还结果。** 告知本步新增能力、仍未做的 UI/
  基准、工作区未提交依赖，以及下一建议小步。不要自动实现 UI。

## 计划自检与执行交接

设计覆盖：§1–3→Task 1/4/6；§4–5.1→Task 1；§5.2–6→Task 2/3；
§7→Task 3/4/5；§8→Task 4/6；§9→各任务断言及 Task 6；§10→本交接。
计划审阅阶段只检查文字、接口和覆盖关系；实际实施进度以上方状态及任务勾选为准。

用户已审阅计划并选择第一种方式：

- **当前会话逐步实施（本计划建议）**：主代理顺序完成每个任务，解释与验证后
  才进下一步，最后由独立审查者检查整体；接口依赖紧密，减少交接偏差。
- **逐任务独立实施与审查**：每个任务由独立子代理实现，再由新的审查者检查，
  通过后进入下一任务；检查更密集，但时间与上下文消耗更高。

按用户逐步学习、不一次生成全部代码的要求，每轮只交付一个已确认的小步。后续任务
保留同一计划与执行记录；范围仅限组合核心，不能扩大为 UI、自动交易或后续阶段。
