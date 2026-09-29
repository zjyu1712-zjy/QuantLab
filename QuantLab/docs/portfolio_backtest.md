# 多 ETF 组合回测：逐步实现说明

## 当前进度

已实现第 1 步“固定调仓时间表与数值目标”、第 2 步“费用与独立买入预算”、
第 3 步“现金、订单与每日估值”、第 4 步“研究快照接入与运行身份”，
以及第 5 步“完整资金曲线的绩效分析”。
公开 Python 入口已能在已验证快照上运行内置因子，返回逐笔账本、持仓、
现金和每日资产曲线，并保留本次输入与规则。

组合绩效已可通过 Python 入口计算；目前没有新增页面，也不自动保存回测档案。
筛选层的 `PENDING` 仍仅代表计划；引擎执行流程完成后改为 `EXECUTED`，
实际买了多少须查看订单/成交表，不能把流程完成等同于所有目标都买足。
现有因子页面的每日 Top N 预览仍可使用，但没有自动接入此引擎。

第 6 步的 53 项端到端验收已通过；全项目回归与整体独立复查结果见文末。
本轮只验收组合回测 Python 核心，不等于第二阶段全部完成，也不涉及自动交易。

## 第 1 步的代码分别做什么

- `src/quantlab/portfolio/types.py`：定义组合配置、结构化错误和调仓安排。
  对 N、调仓间隔、排序方向及嵌套交易参数做校验，不静默转换错误输入。
- `src/quantlab/portfolio/selection.py`：核对因子表，安排信号日期，
  用原始数值筛选 ETF，返回决策表和目标表；不读取行情文件或计算成交。
- `tests/conftest.py`：追加隔离的合成行情夹具，通过真实导入和研究集合
  构建流程生成测试输入；不读取或改动个人行情、交易记录。
- `tests/test_portfolio_selection.py`：验证时间安排、筛选、缺失状态、
  并列值、配置错误，以及输入数据不会被修改。

## 先有历史，再产生目标

设因子回看期数为 L，调仓间隔为 K，研究日期轴从位置 0 开始。
固定信号位置为 `L, L + K, L + 2*K, ...`，不是按自然日或每月调仓。

例如使用 20 期动量、每 20 期调仓：

- 位置 0～19 为热身，尚未生成调仓决策。
- 位置 20（第 21 个研究日期）收盘后，第一次按因子值选择 ETF。
- 若有下一日期，计划在位置 21 开盘执行，不能在信号当日开盘成交。
- 下一次信号固定在位置 40。即使位置 20 因数据不足跳过，也不推迟到
  “第一次有足够数据之日”重新计时。

已支持的动量和波动率因子都需要 L+1 个连续窗口价格点；有效 ETF 不足
N 只时，没有目标配置。研究日期轴由现有研究集合提供，不自动补成官方
交易日历，也不因某个日期有缺失而压缩日期轴。

## 决策状态

| 状态 | 含义 | 是否有目标行 |
| --- | --- | --- |
| `PENDING` | 有效 ETF 足够，且存在下一研究日期，等待后续引擎执行 | 恰好 N 行 |
| `SKIPPED_INSUFFICIENT_FACTORS` | 有效 ETF 不足 N，跳过本次调仓 | 无 |
| `EXPIRED` | 有效 ETF 足够，但信号已经在最后一个研究日期 | 恰好 N 行，仅供解释 |

没有下一研究日期时，`execution_date` 为 `NaT`。最后一天若也不足 N，
状态仍是“有效因子不足”，不覆盖为“过期”。

“跳过”不是清仓指令。后续引擎应保留原现金和持仓，并照常对持仓估值。
初始资金也会在热身及未成功建仓期间保持现金；第 1 步只记录安排，
尚未计算这条现金路径。默认初始现金 100,000 元来自既有交易配置。

## 排序和目标权重

- 动量默认从大到小，选择过去价格动量较强的 ETF。
- 波动率默认从小到大，选择过去价格波动较小的 ETF。
- `ascending=True/False` 可明确覆盖排序方向；低波动不意味着未来必然
  低风险，高动量也不意味着未来必然上涨。
- 用未经显示舍入的原始因子值排序。有效的 0 和负值也参与，缺失及
  无效值不参与，不将其填成 0。
- 原始值相同时，按六位 ETF 代码升序确定先后，并且恰好选 N 只。
  若第 N 与第 N+1 只原值相同，`cutoff_tie=True` 并记录提示；代码顺序
  只是确定性规则，不表示投资优劣。
- 每只目标权重为未经舍入的 `1/N`。这是配置目标，不是已经成交后的
  实际权重；后续整手约束、费用和余款都会使实际权重产生偏差。

## 内部调用示例

以下片段用于已有 `preview`（研究集合快照）的 Python 代码中，
**不是直接粘贴到终端的命令**。本段只解释筛选模块；正式组合研究请使用
下文第 4 步的公开入口。

```python
from quantlab.factors import run_research_factor
from quantlab.portfolio import PortfolioConfig
from quantlab.portfolio.selection import build_rebalance_schedule

# 用同一份研究快照计算 20 期动量。
factor = run_research_factor(preview, "momentum_20")

# 从至少两只 ETF 的研究集合中选 2 只，每隔 20 个研究期产生目标。
config = PortfolioConfig(top_n=2, rebalance_interval=20)
schedule = build_rebalance_schedule(factor, config=config)

print(schedule.decisions)  # 信号日期、计划执行日期、状态及原因
print(schedule.targets)    # 选中的 ETF、原始因子值和目标权重
```

`build_rebalance_schedule` 是内部数值模块，不从 `quantlab.portfolio`
顶层导出。它校验完整、有序的日期×代码网格、数值类型和有效状态，
但不认证因子表是否被人为改写，也不重算输入指纹；这些保障属于第 4 步
研究快照运行入口。不要把手工改写后的因子结果当作可追溯研究记录。

返回的表与输入独立，调用不会改动原表或依赖其行索引标签。
`RebalanceSchedule` 虽然是冻结容器，其中的 DataFrame 仍可修改；
修改结果不影响输入，但也不代表修改后的安排经过重新验证。

## 验证方法

在 QuantLab 项目目录和现有虚拟环境下运行：

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_selection.py tests/test_ui_factors.py -q
```

测试涵盖 20/60/120 期回看、1/20 期间隔、热身不足、末日期过期、
缺失窗口后保持固定调仓日期、全无效不生成清仓目标、精确同值和显示
舍入差异，以及与现有每日 Top N 预览的数值一致性。

这些测试验证程序规则，不证明策略具有盈利能力。第 5 步已有单独的组合绩效
测试；完整端到端验收仍属后续任务，不能由筛选层测试代替。

## 第 2 步：费用与买入预算

新增 `src/quantlab/portfolio/execution.py`，只有纯计算函数：不读取文件、
不修改现金和持仓、不连接券商，也不会生成真实订单。

| 函数 | 作用 |
| --- | --- |
| `commission` | 计算一笔成交的佣金：比例佣金与最低佣金取较大者；零成交额佣金为 0 |
| `execution_price` | 买价为开盘价 × (1 + 滑点率)，卖价为开盘价 × (1 − 滑点率) |
| `affordable_lot_quantity` | 在独立预算内，寻找不超过申请上限且能支付佣金的最大整手份额 |
| `allocate_buys` | 一次性分好各只 ETF 的含费预算，再分别计算可买数量，返回含零成交的代码字典 |

`affordable_lot_quantity` 的价格参数已经包含滑点，函数不会再加一次滑点。
买入不收卖出税；卖出税和逐笔卖出净收入已在第 3 步现金引擎中记账。
这里的费用参数沿用已有研究配置，不表示某家券商的实际收费标准。

### 为什么 10 万元会剩下 1,990 元？

假设两只 ETF 价格都是 10 元，每手 100 份；申请各买 5,000 份。
比例佣金和滑点为 0，每笔最低佣金为 5 元，初始可用现金为 100,000 元。

1. 每只完整需求为 `5000 × 10 + 5 = 50005` 元，两笔合计 100,010 元，
   超过可用现金，不能直接各买 5,000 份。
2. 两笔完整需求相同，所以同时获得 50,000 元的独立含费预算。
3. 每只最多买 4,900 份，支出 `4900 × 10 + 5 = 49005` 元。
4. 两笔共支出 98,010 元，余款 `100000 − 98010 = 1990` 元留作现金。

即使余款还能支持某只多买一手，也不二次分配，避免代码排在前面的 ETF
优先用完余款。因此目标等权，不意味着扣费后的实际权重精确等权或满仓。

### 分配和输入规则

对正申请份额 d、含滑点买价 p，完整需求为：

`need = d × p + max(d × p × 佣金率, 最低佣金)`

现金足够时买足全部申请；不足时，各自预算为
`cash × need / 全部 need 之和`。先确定所有预算，再按整手买入；输入
字典的先后顺序不影响结果，输出按 ETF 代码排序。零申请无需开盘价，
实际可买数量为零时不收最低佣金。

- 独立求数量函数允许最大份额不是整手，按不超过上限的整手寻找。
- 多资产分配函数要求申请已经是整手差额；不默默替调用方修正非整手申请。
- 正申请必须有正的有限开盘价；金额、费率、数量和方向均在入口校验。
- 大申请量使用整数手数二分查找，不是一手一手往下减，因此不会因申请
  了极大份额而进入超长循环。
- 不把价格或费用四舍五入到分，不用相对或绝对容差放宽买入预算。
  余额归整容差不是可借用的资金；哪怕只差一点，也不能多买一手。
- 溢出、非零滑点/佣金被数值精度完全吞掉、最终份额无法精确参与金额
  运算，或分配舍入使合计支出超过现金时，抛出 `PortfolioBacktestError`
  （`reason="numeric_range"`），不返回看似成功的部分配置。

第 3 步引擎对结果表执行 int64 份额范围限制；本预算模块接受 Python 整数
上限，但不是承诺任意巨大的成交数量都能可靠计算。

### Python 示例

以下是 Python 代码，不是终端命令；可在已有 Python 脚本中使用：

```python
from quantlab.config.settings import BacktestConfig
from quantlab.portfolio.execution import allocate_buys

costs = BacktestConfig(
    commission_rate=0, minimum_commission=5,
    slippage_rate=0, lot_size=100,
)
quantities = allocate_buys(
    100000,
    requests={"510300": 5000, "159915": 5000},
    opens={"510300": 10, "159915": 10},
    config=costs,
)
print(quantities)  # {'159915': 4900, '510300': 4900}
```

第 2 步测试命令（在项目目录运行）：

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_execution.py -q
```

测试覆盖最低佣金交界、买卖滑点方向、现金不足、含费需求分配、余款
不再分配、字典顺序独立、大数量计算、非法输入及数值范围保护。

## 第 3 步：现金、订单与每日估值

新增 `src/quantlab/portfolio/engine.py`，提供内部函数 `simulate_portfolio`。
`types.py` 新增 `PortfolioBacktestResult`，并集中定义六张表的字段及类型。
`selection.py` 仅改为引用这套统一结构，原有筛选规则不变。

引擎每天完成以下工作：

1. 查找应该在当天开盘执行的前日配置，检查原持仓及目标 ETF 的开盘价。
2. 根据开盘交易前总资产计算一次整手目标；先卖出多余份额或退出的资产。
3. 对待买差额同时分配含费预算，再买入。每侧按代码升序记录，不抢占预算。
4. 检查实际成交资产是否为零成交量；没有数量差额或实际买不到一手时不误报。
5. 用当日收盘价估值持仓，核对现金、份额与总资产，完成后才接受当天记录。

目标不按扣费后的总资产反复重算；保留的 ETF 只交易数量差额，不全部卖掉
再买回。信号日不开仓，热身阶段的全现金也保留在资金曲线中；期末不强制平仓。

### 三日手算账本

假设初始资金 1,000 元，每手 10 份，佣金、税、滑点全为 0。
A=510300，B=159915，C=512000；每次选两只，目标各占一半。

- D0 收盘选择 A/B；D1 开盘价分别为 10/20 元。
  因整手限制，买 A 50 份、B 20 份，余款 100 元。
- D1 收盘价变为 A=11、B=18；当日收盘选择 B/C。
- D2 开盘价 A=12、B=20、C=5，交易前资产为 1,100 元。
  卖掉 A 的 50 份，保留 B 的 20 份，买入 C 的 110 份。
  D2 收盘价与开盘价相同。

| 日期 | 期末现金 | A 份额 | B 份额 | C 份额 | 持仓市值 | 总资产 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 1000 | 0 | 0 | 0 | 0 | 1000 |
| D1 | 100 | 50 | 20 | 0 | 910 | 1010 |
| D2 | 150 | 0 | 20 | 110 | 950 | 1100 |

D1 的成交记录先买 B，现金变为 600，再买 A，变为 100；D2 先卖 A，
现金变为 700，再买 C，变为 150。订单和成交按真实模拟顺序保存，不能
最后按代码全局重排，否则“成交后现金”就无法顺着行核对。

### 返回哪些内容

- `decisions`：计划、跳过、过期与执行流程状态。
- `targets`：信号日的数值目标，按日期/代码输出独立副本。
- `orders`：申请量、实际量、未成交量和原因；区分全部成交、部分成交与拒单。
- `trades`：只有正成交数量的记录，含佣金、卖出税、滑点金额、成交后现金及份额。
- `positions`：每天每只 ETF 的份额、市值和实际权重，包括空仓代码。
- `equity_curve`：每日现金、持仓市值、总资产、净值、累计收益、累计佣金税费及累计滑点。

拒单没有成交 ID，也不收费。滑点已经体现在成交价，现金不重复扣除；
累计滑点单独展示，与累计佣金税费分列。数值目标不等于实际持仓权重。

### 缺失与失败边界

跳过调仓时保留持仓并继续估值。原持仓缺收盘价仍须失败；空仓资产缺价
则保留空收盘价，市值和权重为 0，不做 `0 × NaN` 运算。

缺必需开盘价、持仓缺收盘价、实际成交资产零成交量、部分损坏的 OHLCV 行、
负卖出净收入、数值溢出或数量越界都会抛出结构化错误。运行时错误带日期、
相关代码与原因；整个运行失败，不返回部分资金曲线，也不修改输入。

补充的保守限制：如果费用等使总资产恰好耗尽为 0，权重和后续指标无法按
当前口径定义，报 `non_positive_equity`，不构造虚假权重。微小的正现金不会
被抹成 0；`1e-9` 容差仅处理运算后的微小负余额，不能用来多买一手。

正成交量不证明能够按开盘价成交。复权价下的份额、静态 ETF 池、研究日期轴
等原有限制仍然存在，本引擎不是实盘可成交性证明。

### 内部调用与验收

在已有 `preview` 研究集合的 Python 脚本中：

```python
from quantlab.factors import run_research_factor
from quantlab.portfolio import PortfolioConfig
from quantlab.portfolio.selection import build_rebalance_schedule
from quantlab.portfolio.engine import simulate_portfolio

config = PortfolioConfig(top_n=2)
factor = run_research_factor(preview, "momentum_20")
schedule = build_rebalance_schedule(factor, config=config)
result = simulate_portfolio(
    preview.alignment.data, preview.alignment.calendar, schedule, config=config,
)
print(result.trades)
print(result.equity_curve)
```

该函数仍是内部记账入口，不认证手动改写的安排是否来自因子。
直接调用内部引擎时，`metadata/input_data/factor_data` 为 `None`；
请使用第 4 步公开研究入口绑定输入快照、参数和运行身份。
内部记账结果不等于可追溯研究档案。

测试命令（在项目目录执行）：

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_engine.py tests/test_portfolio_execution.py -q
```

合成测试包括独立手算账本、真实因子热身后的执行日期、只交易差额、先卖后买、
费用/税/滑点对账、缺价与零成交量、输入不变及逐日资产/权重恒等式。

## 第 4 步：研究快照接入与运行身份

新增 `src/quantlab/portfolio/research.py`，用一个公开函数
`run_factor_portfolio` 串起已有模块。`__init__.py` 导出该函数及组合结果类型，
不把内部手动安排或外部因子表开放成正式研究入口。

### 代码依次做什么

1. **隔离并验证输入。** 复制研究预览和交易配置，由 `run_research_factor`
   验证原有行情、来源、日历、覆盖报告、警告与指纹，再用新建的内置注册表计算。
   内容或身份损坏时拒绝，不重新生成指纹来把改动“修好”。
2. **规范自己的行索引。** 行标签不是数据身份；重复、不连续、多级且与
   `date/symbol` 同名的索引仍可使用。只清理副本，不改调用方表格。
3. **产生调仓安排并核对源码身份。** 调用已有筛选模块；复用第一阶段
   `create_metadata` 的运行编号、时间、环境与源码加载基线检查，不套用其单 ETF 身份格式。
4. **完成模拟。** 调用已验证的现金/持仓引擎；缺价、零成交量等错误保留日期、
   ETF 代码及原因。失败不返回截断成果。
5. **绑定结果身份。** 使用独立的 `portfolio-run-v1` 格式，附上行情、因子表
   的内存副本和内容指纹。整个公开入口不回读源行情、不联网、不写档案。

### 公开调用示例

下列是放入 Python 脚本的代码，不是逐行粘贴到终端的命令。`preview` 是
已经由研究集合模块构建或载入的有效 `ResearchSetPreview`，至少含两只 ETF；
不要求它已保存，也不要求其原来源文件仍在原路径。

```python
from quantlab.portfolio import PortfolioConfig, run_factor_portfolio

# 从研究集合中选 2 只，默认按动量从大到小，每 20 个研究期调仓。
config = PortfolioConfig(top_n=2, rebalance_interval=20)
result = run_factor_portfolio(preview, factor_name="momentum_20", config=config)

print(result.trades)                  # 真实模拟成交及每笔费用
print(result.equity_curve)            # 含完整热身期的每日总资产
print(result.metadata["run_id"])      # 每次运行的独立编号
print(result.metadata["strategy"])    # 本次实际配置、方向和时间规则
print(result.metadata["warnings"])    # 解释结果前应阅读的限制
```

支持 `momentum_20/60/120`、`volatility_20/60` 五个内置因子。
这里的名称是注册名，不接受公式文本；也没有外部 `registry` 或 `factor_data`
参数。动量默认降序、波动率默认升序，`ascending` 可显式覆盖。

### 运行记录包含什么

| 信息 | 记录位置与作用 |
| --- | --- |
| 运行身份 | `run_id/run_at` 为独立编号和带时区时间；`status=complete` 只在完整模拟成功后返回 |
| 规则和资本 | `strategy` 保存规则版本、完整配置、实际排序方向、回看期 L、间隔 K 与 `L+k*K` 锚点 |
| 因子定义 | `factor_definition` 保存公式说明、版本、所需字段、窗口、方向、缺失规则与可用时点 |
| 输入内容 | `data` 保存预览、对齐、行情和因子指纹、来源版本、区间、价格口径及完整日期轴声明 |
| 源码与环境 | `code_sha256/environment` 包含实际加载基线的源码指纹、Python 和核心数值库版本 |
| 适用限制 | `warnings` 保留上游警告及静态池、复权模型单位、成交假设、取整与目标权重偏离等限制 |

`result.input_data` 保留八列行情（含 `has_observation`）；
`result.factor_data` 保留数值、有效标记与不可用原因。它们与调用方、其他运行
不共享可修改内容。来源信息只用既有白名单，并排除 `processed_path`；
不把机器行情路径或凭证字段装进运行元数据。

内容指纹帮助确认“数据有没有变”，来源/版本是保留声明，都不证明数据在
历史当时已经可得。固定数据可能包含事后修订或复权信息；本记录不是历史真实性
或代码安全认证。修改返回表不会修改原输入，但也不会自动更新指纹，不能把修改后
的结果当作仍然通过验证的原始运行。

如果 Python/页面进程加载模块后磁盘源码改变，再运行会提示**重新启动**；
只重载记录模块不能绕过检查。这避免用新磁盘源码指纹标记旧内存中的函数。
来源文件被移动则不同：只要已载入快照仍有效，就不影响本次纯内存模拟。

仅调用 `run_factor_portfolio` 时，`analytics_config` 为 `None`，表示该运行
尚未附加绩效分析。第 5 步的分析入口会在独立副本中记录实际指标参数。
本步骤也不保存档案、不创建页面，不等于第二阶段全部完成。

### 验证方法

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_research.py tests/test_factor_research.py tests/test_backtest_provenance.py -q
```

测试使用临时合成数据，通过真实 CSV 导入、研究集合、五个内置因子和引擎运行，
检查次期开盘、输入篡改拒绝、行标签兼容、输出隔离、来源文件移动、全现金结果、
结构化执行失败和源码加载身份。源码变动测试只修改临时复制件，不改当前项目。

## 第 5 步：完整资金曲线的绩效分析

新增 `src/quantlab/portfolio/analysis.py`，公开入口是
`analyze_portfolio_result`；`types.py` 新增 `PortfolioPerformance` 与
`PortfolioAnalysis`，分别保存指标/曲线及本次分析对应的独立回测副本。

### 代码依次做什么

1. **检查指标参数。** 每年期数必须为有限正整数，无风险年利率必须有限且
   大于 -100%；布尔值、文本、NaN 和无穷不当成有效参数。默认 252 期、利率 0。
2. **检查完整结果。** 要求 `portfolio-run-v1` 且运行成功，行情和因子快照指纹
   仍一致，资金曲线严格对应完整日期轴；总资产必须有限且为正，起点等于初始资金。
   不悄悄删坏行、补缺值或从首次成交日重新计时。
3. **逐项复用旧公式。** 调用 `analytics/returns.py` 和 `analytics/risk.py`，
   计算简单期收益、标准化净值、累计收益、复合年化、样本波动率、回撤与夏普。
   组合层不复制公式，也不改变第一阶段统一入口原有的报错行为。
4. **区分未定义与错误。** 样本不足或日收益率波动太小的指标返回 `None`，
   原因在 `unavailable_reasons`；数据损坏、算术溢出或非有限计算结果则明确报错，
   不返回部分分析，也不把错误归为“样本不足”。
5. **保留原结果。** 返回独立的 `backtest` 副本，只在其元数据内记录实际
   `analytics_config`；原运行编号、时间、源码指纹和输入身份不被重新生成。
   再次使用其他年化参数分析，也不会改掉上一份分析。

这里校验结构、日期完整性、输入指纹及资产序列，不重放每笔交易来认证结果。
请使用未手动修改的公开研究入口输出；这些检查不是任意手工改写后的真实性认证。

### 为什么“0”和“无法计算”不一样

一个收益率需要两个资产观测值；样本波动率至少需要两个收益率。

| 输入情况 | 累计/年化收益 | 年化波动率 | 夏普比率 | 净值与最大回撤 |
| --- | --- | --- | --- | --- |
| 仅 1 个资产观测 | `None`，没有收益区间 | `None` | `None` | 净值 1，最大回撤 0 |
| 2 个资产观测 | 可以计算 | `None`，仅一个收益率 | `None` | 可以计算 |
| 至少 3 个观测、始终全现金 | 0 | 0 | `None`，波动为零 | 净值始终 1，最大回撤 0 |
| 至少 3 个观测、收益有足够波动 | 可以计算 | 可以计算 | 可以计算 | 可以计算 |

夏普沿用既有阈值：**日收益率样本标准差不高于 `1e-15` 时未定义**，
比较的不是年化波动率。固定正收益也可能没有足够波动，不能把夏普写成无穷大。
最大回撤沿用原指标的符号：例如 `-0.12` 表示相对历史最高净值下跌 12%。

只有一个资产观测时，`daily_returns` 是带日期索引的空浮点序列；
其他情况下包含从第二个研究日期开始的相邻期收益率，不给首日伪造一个 0 收益。
期数按研究日期轴而非自然日计，日期轴与复权数据的原有限制仍保留。

### 费用和热身期没有丢失

合成测试中，初始 100,000 元，前 21 个资产观测都为空仓；第 22 个日期首次
建仓并支付 10 元费用，总资产变为 99,990 元；次日变为 100,440 元。

- 累计收益为 0.44%，不是从 99,990 元重新起算。
- 首次交易期收益为 -0.01%，这笔费用在净值与回撤中可见。
- 年化使用全部 22 个收益区间：`1.0044 ** (252 / 22) - 1`。

### 调用和查看原因

以下放入已有 `result` 的 Python 脚本中，不是终端命令：

```python
from quantlab.config.settings import AnalyticsConfig
from quantlab.portfolio import analyze_portfolio_result

analysis = analyze_portfolio_result(
    result,
    analytics_config=AnalyticsConfig(periods_per_year=252, annual_risk_free_rate=0.0),
)
p = analysis.performance
print(p.cumulative_return, p.annualized_return)
print(p.annualized_volatility, p.max_drawdown)
if p.sharpe_ratio is None:
    print(p.unavailable_reasons["sharpe_ratio"])
else:
    print(p.sharpe_ratio)

print(p.net_value)          # 完整研究区间的净值曲线
print(p.drawdown)           # 回撤曲线
print(p.unavailable_reasons)  # 只列未定义指标及原因
```

结果仍在内存中，不自动保存或显示到页面；不包含胜率、盈亏比、组合基准或超额收益。
完整链路验收和整体复查见第 6 步，不在本计划内扩大到新策略。

### 验证方法

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_portfolio_analysis.py tests/test_performance.py tests/test_backtest_analysis.py -q
```

测试覆盖 1/2/3 个观测、全现金、手算费用与完整热身、旧公式一致性、非零/负无风险利率、
夏普阈值、结果隔离及再次分析、缺失/重排日期、损坏快照、非法参数和有限输入下的指标溢出。

## 第 6 步：完整链路验收

新增 `tests/test_portfolio_acceptance.py`，不新增业务入口。测试的作用是把
前五步真正连接起来，验证模块分别正确时，连接后仍保持同样的规则。

1. **脱离源文件重放。** 用真实 CSV 导入、构建并保存 65 日期研究集合，
   将临时源目录改名后重读快照、回测、分析。八张业务表与保存前运行一致；
   运行编号可以不同，但输入和业务结果不变。快照文件字节不变，也没有自动生成档案。
2. **检查时序隔离。** 对 20 期动量和波动率分别选 1/2 只，每期调仓，
   只改变位置 30 起的未来价格。严格早于该时点的决策、目标、订单、成交、
   持仓、资产与因子均不变。这个检查不证明历史数据当时可得或不存在幸存者偏差。
3. **区分日期轴。** 两只 ETF 在热身期同一天无记录时，用户提供的日期轴
   仍保留这一天的现金估值；行情并集轴会遗漏该日，两者不能当成同一输入。
4. **检查旧模块边界。** 解析第一阶段数据、指标、策略、回测和配置模块的
   导入语句，确保没有反向引入 `portfolio` 或 `factors`。
5. **独立对账。** 45 种组合覆盖五个因子、默认/正/反方向、不同 N/K、
   手数、佣金下限、卖出税与滑点。逐笔用成交额、佣金和税重算现金与份额，
   每日检查市值、总资产、权重、累计费用及完整区间收益；不复用引擎内部记账函数。

这些都是合成数据的自动化验收，不是策略有效性、真实可成交性或盈利证明。

### 从已保存研究集合开始

以下代码放在 Python 脚本内运行；`set_id` 替换为已保存研究集合的实际编号。
该集合需要至少两只 ETF；路径也应指向你实际保存集合的目录。

```python
from pathlib import Path
from quantlab.data.research_storage import load_research_set
from quantlab.portfolio import (
    PortfolioBacktestError, PortfolioConfig,
    analyze_portfolio_result, run_factor_portfolio,
)

preview = load_research_set(
    Path("data/research_sets"), "替换为实际研究集合编号",
).preview

try:
    result = run_factor_portfolio(
        preview, factor_name="momentum_20",
        config=PortfolioConfig(top_n=2, rebalance_interval=20),
    )
    analysis = analyze_portfolio_result(result)
except PortfolioBacktestError as error:
    # 缺少必要价格等错误时，没有完整期间的成功结果。
    print(error.reason, error.date, error.symbols, str(error))
else:
    print(analysis.performance.cumulative_return)
    print(analysis.performance.unavailable_reasons)
    print(analysis.performance.net_value)
    print(result.orders)   # 包括买不足或买不到的申请
    print(result.trades)   # 只有实际模拟成交及费用
```

阅读结果时，先看 `decisions` 的跳过/过期原因和 `metadata["warnings"]`，
再看目标与实际持仓差异，最后查看完整资产曲线及绩效。不能仅凭一个收益数字判断策略。

### 验收范围与下一步

本轮完成单因子 Top N、固定间隔、下一研究期开盘、只做多不融资的组合核心。
尚未实现组合界面、基准/超额收益、组合档案和下载、胜率与盈亏匹配、稳健性检验。
下一小步建议单独设计 Streamlit 组合研究入口，不自动扩大本次交付。

验证针对当前工作区，其中仍有此前未提交的实现（包括既有
`backtest/provenance.py`）。本轮没有替用户提交这些文件，不能把单独检出的
本轮提交描述为自包含的可发布版本。没有运行浏览器验收或真实外部行情接口。

### 本次执行记录（2026-09-30）

- 开始前全项目基线：1291 项通过、2 项外部接口测试跳过。
- 新增端到端验收：52 项通过，用时 10.71 秒；现有实现已满足这些断言，未人为制造失败。
- 修复前全项目回归：1343 项通过、2 项跳过，用时 49.86 秒。跳过的是默认不联网的
  AkShare 检查，以及需要显式开关和凭证的 iFinD 集成测试。
- 独立审查覆盖六步实现和五项重点，单独执行 453 项组合测试及全项目回归。
  发现 1 项重要问题：极大现金余额可能吞掉极小的非零交易现金流。
- 修复先观察失败：买入、卖出、真实 CSV 公开入口共 3 项失败；净收入合法为零的
  对照通过。随后在现金更新边界增加 `numeric_range` 拒绝，并保留日期和 ETF 代码。
  不改变一般交易规则，也不声称支持任意精度金额。
- 修复后引擎和端到端共 111 项通过（其中端到端 53 项）；最终全项目
  **1347 项通过、2 项外部测试跳过，用时 55.67 秒**。依赖无冲突，差异无空白错误。
  重要问题已通过失败→修复→通过及全回归验证，没有另行宣称修复后的第二次独立审查。
- 延后的小项：第 1 步末尾还保留“端到端验收仍属后续任务”的历史进度句，
  本轮功能修复未处理这处文案；当前状态以本节的实际验收记录为准。

独立审查未认定为本轮新增功能的五类事项，已由实施者明确保留边界：
独立干净检出的可发布性；界面/外部行情/基准/FIFO/档案等未实施或未验收能力；
历史可得性、官方日历及真实成交证明；手改结果/内部手动安排的独立认证；
零总资产路径继续成功运行。它们分别意味着仍需整理依赖、补后续功能、谨慎解释
回测、使用未经手改的公开结果，以及接受总资产耗尽时停止分析，不能视为已验证能力。
