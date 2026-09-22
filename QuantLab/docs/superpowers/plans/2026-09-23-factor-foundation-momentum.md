# 轻量因子接口与 20 期动量 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有多 ETF 研究集合上计算可追溯的 `momentum_20`，逐行解释不可用原因，不改动原行情或既有单 ETF 功能。

**Architecture:** 先建立因子定义、数值/状态结果和显式注册接口，再实现严格 21 条观测窗口的 20 期价格动量。最后用一个纯内存研究服务连接现有 `ResearchSetPreview` 校验、逐 ETF 计算、按键合并及来源说明；不更改研究快照存储格式。

**Tech Stack:** 现有 Python 3.12、pandas、NumPy、pytest；端到端测试复用现有 CSV 导入及 PyArrow 快照存储。不新增依赖，不联网。

**Spec:** `docs/superpowers/specs/2026-09-23-factor-foundation-momentum-design.md`（2026-09-23 用户已确认；执行者先完整阅读）。

## Global Constraints

- “用户要求逐步学习、解释每一步，不能一次实现整个第二阶段。”
- “使用已校验研究集合，不修改原行情或快照。”
- “按研究集合日期轴计数，不按自然日、不跳过缺失日期。”
- “当前与此前 20 期共 21 条记录必须全部有效；否则留空。”
- “先实现计算、测试和说明，不增加界面、排名或组合回测。”
- “不新增依赖；复用现有 Python、pandas、NumPy 和测试环境。”
- “保留当前项目目录与分支，保护既有未提交的第一阶段和 iFinD 修改。”
- “本步固定周期为 20，不开放可变窗口或自动生成 60/120 期等其他因子。”
- “不回读原文件或隐式扩大范围。”
- “全部使用临时合成数据，不联网、不读取私人行情或凭证。”
- 公式 `close[t] / close[t-20] - 1`；名称 `momentum_20`；版本 `1.0`；`strict_window`；`after_close`。
- 不执行用户文本公式，不注册共享可变全局表，不在导入包时运行研究。
- 保留 `BaseFactor.compute(DataFrame) -> Series` 和 `FactorRegistry.register/get/names` 调用形式。
- 本计划中的代码是实施指导，不是已落地代码或测试通过记录；只在对应任务获准执行时写入产品文件。

## Review Focus

以下五类易漏条件各有归属任务和具体测试，不能只验算正常涨幅：

1. Python 的 `bool` 可被当作整数、可空布尔列可能含 NA：周期拒绝布尔值，观测列拒绝 NA，不能把未知当成未观测（Task 1、2）。
2. pandas 日期单位、时区、盘中时间及非连续索引：只接受规范纳秒日线，保留合法单 ETF 行标签；研究入口对副本重置标签（Task 2、3）。
3. 端点存在但窗口中间缺失、或某只 ETF 全缺失：留空并区分热身和缺失，不能仅检查首尾或删除资产（Task 2、3）。
4. 两个价格均有限但除法上溢、下溢或减 1 舍入成 -1：输出 `numeric_range` 和 NaN，不报告无穷或虚假 -100%（Task 2）。
5. 因子实现返回错序/缺行/错状态，或修改传入的计算表：服务拒绝不合约输出、隔离输入和用于合并的键，不静默错配价格与日期（Task 3）。

---

## 工作目录、基线与边界

项目目录：`/Users/zhoujiayu/Documents/量化/QuantLab`；Git 根目录在其上一级。
以下文件路径相对项目目录，命令在项目目录运行。用户已选用当前目录，
本计划不新建工作树、不切分支、不推送或创建 PR。

当前已有大量不属于本任务的未提交修改。执行前记录工作树并运行基线：

```bash
git status --short
PYTHONPATH=src .venv/bin/python -m pytest -q -rs
.venv/bin/python -m pip check
```

`PYTHONPATH=src` 显式提供源码路径，不依赖当前尚未提交的 `pytest.ini`。
记录本次实际通过、跳过及失败信息；历史测试数量不是本次证据。遇到基线
失败先诊断，不修改无关功能来凑通过数。不得 `git add .`，不得将未归属
本任务的文件一并提交。三项任务顺序执行，每项完成后解释作用和结果；
用户希望分轮学习时在任务边界交还，不一次实现整个第二阶段。

## 文件职责及接口地图

| 文件 | 本步职责 |
|---|---|
| `src/quantlab/factors/base.py` | `FactorDefinition`、`FactorEvaluation`、`BaseFactor` |
| `src/quantlab/factors/registry.py` | 类型/名称校验，显式注册及默认注册表工厂 |
| `src/quantlab/factors/momentum.py` | 单 ETF 规范输入验证及 20 期动量 |
| `src/quantlab/factors/research.py` | `FactorRunResult`、研究入口、输出契约验证、追溯与警告 |
| `src/quantlab/factors/__init__.py` | 上述公开接口的明确导出，无运行副作用 |
| `tests/test_factor_contracts.py` | 定义、注册、旧数值调用方式 |
| `tests/test_momentum_factor.py` | 手算、输入类型、窗口、数值边界、时间隔离 |
| `tests/test_factor_research.py` | 合成导入夹具、跨 ETF、来源、坏结果和真实快照链路 |
| `docs/factors.md` | 学习说明、用法、状态含义、价格与时点限制、验收记录 |

复用而不改动：`data/research_sets.py` 的 `ResearchSetPreview`、
`validate_research_preview`、`SOURCE_FIELDS`；`data/calendar.py` 的
`CalendarAxis`；`data/research_storage.py` 的保存/重读接口。
不改第一阶段页面、Provider、回测、依赖清单及已存在脏改动的 README。

依赖顺序：Task 1 接口与注册 → Task 2 具体公式与默认工厂 → Task 3 研究服务。
不并行实现依赖同一接口的三项任务；只允许独立、只读检查并行。

## Task 1：建立统一因子接口和安全注册

**作用：** 给因子一个可核对的“说明书”，并让调用方同时取得数值和不可用原因。

**Files:**

- Modify: `src/quantlab/factors/base.py` 全部占位接口。
- Modify: `src/quantlab/factors/registry.py` 的 `register/get`。
- Modify: `src/quantlab/factors/__init__.py` 的文档及导出。
- Create: `tests/test_factor_contracts.py`。

**Interfaces:**

- Consumes: 现有 `register(type)`、`get(name)`、`names()` 方法名。
- Produces: 下面三个类型；`compute` 为具体方法，`evaluate` 为唯一抽象计算方法。
- Produces: `FactorRegistry.get(name: str) -> type[BaseFactor]`，未知名字抛中文 `KeyError`；坏注册项抛中文 `ValueError`。
- 本任务不添加默认因子工厂，避免依赖尚未存在的 `momentum.py`。

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
import pandas as pd

@dataclass(frozen=True)
class FactorDefinition:
    name: str
    description: str
    formula: str
    version: str
    required_fields: tuple[str, ...]
    lookback_periods: int
    window_observations: int
    direction: str
    missing_policy: str
    availability: str

@dataclass(frozen=True)
class FactorEvaluation:
    values: pd.Series
    status: pd.Series

class BaseFactor(ABC):
    name: ClassVar[str]
    definition: ClassVar[FactorDefinition]

    def compute(self, market_data: pd.DataFrame) -> pd.Series:
        return self.evaluate(market_data).values.copy(deep=True)

    @abstractmethod
    def evaluate(self, market_data: pd.DataFrame) -> FactorEvaluation:
        raise NotImplementedError
```

- [x] **Step 1：先写接口和注册红灯测试。** 测试文件先引入尚不存在的类型；
  以下合成因子仅作接口测试，绝不加入产品默认表。

```python
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from quantlab.factors import BaseFactor, FactorDefinition, FactorEvaluation, FactorRegistry

DEFINITION = FactorDefinition(
    name="test_factor", description="接口测试", formula="constant 0",
    version="1.0", required_fields=("close", "has_observation"),
    lookback_periods=20, window_observations=21,
    direction="higher_is_stronger", missing_policy="strict_window", availability="after_close",
)

class TestOnlyFactor(BaseFactor):
    __test__ = False
    name = "test_factor"
    definition = DEFINITION

    def evaluate(self, market_data):
        return FactorEvaluation(
            pd.Series(0.0, index=market_data.index, dtype="float64"),
            pd.Series("valid", index=market_data.index, dtype="string"),
        )

def test_compute_keeps_series_interface_and_index():
    frame = pd.DataFrame({"close": [10.0, 11.0]}, index=pd.Index([3, 9], name="row"))
    factor = TestOnlyFactor()
    actual = factor.compute(frame)
    pd.testing.assert_series_equal(actual, factor.evaluate(frame).values)
    assert actual.dtype == np.dtype("float64")
    assert actual.index.identical(frame.index)

def test_registry_is_explicit_independent_and_duplicate_safe():
    left, right = FactorRegistry(), FactorRegistry()
    assert left.names() == right.names() == ()
    left.register(TestOnlyFactor)
    assert left.get("test_factor") is TestOnlyFactor
    assert left.names() == ("test_factor",) and right.names() == ()
    with pytest.raises(ValueError, match="重复|已注册"):
        left.register(TestOnlyFactor)
    with pytest.raises(KeyError, match="未注册"):
        right.get("test_factor")

@pytest.mark.parametrize("bad", [None, 123, object, BaseFactor, TestOnlyFactor()])
def test_registry_rejects_nonconcrete_factor_types(bad):
    with pytest.raises(ValueError, match="因子|类型|抽象"):
        FactorRegistry().register(bad)

@pytest.mark.parametrize("name", [None, 1, "", " ", " test_factor "])
def test_registry_rejects_invalid_class_names(name):
    bad = type("Bad", (TestOnlyFactor,), {"name": name})
    with pytest.raises(ValueError, match="名称"):
        FactorRegistry().register(bad)

def test_registry_rejects_mismatched_or_missing_definition():
    for definition in (None, replace(DEFINITION, name="different")):
        bad = type("Bad", (TestOnlyFactor,), {"definition": definition})
        with pytest.raises(ValueError, match="定义|名称"):
            FactorRegistry().register(bad)

@pytest.mark.parametrize(("field", "value"), [
    ("name", ""), ("description", " "), ("formula", None), ("version", 1),
    ("required_fields", []), ("required_fields", ("close", "close")),
    ("required_fields", ("",)), ("lookback_periods", True),
    ("lookback_periods", 0), ("window_observations", False),
    ("window_observations", 20), ("direction", ""),
    ("missing_policy", ""), ("availability", ""),
])
def test_definition_rejects_invalid_metadata(field, value):
    with pytest.raises(ValueError):
        replace(DEFINITION, **{field: value})
```

- [x] **Step 2：运行红灯。**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_factor_contracts.py -q
```

预期首次失败于尚未导出的 `FactorDefinition`，而不是路径或依赖错误。

- [x] **Step 3：实现类型与注册校验。** 在上述定义中添加 `__post_init__`：
  七个文本字段为非空、无两端空白的字符串；`required_fields` 为非空、
  唯一、无空白项的字符串元组；两个周期字段必须 `type(value) is int`，
  lookback 大于 0；当前 `strict_window` 要求 `window_observations == lookback_periods + 1`。
  定义中不自动写时间、路径或环境信息。检查核心写法：

```python
# FactorDefinition.__post_init__ 中的周期及窗口约束。
for label, value in (("回看周期", self.lookback_periods),
                     ("窗口观测数", self.window_observations)):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label}必须为正整数，不能为布尔值")
if self.missing_policy == "strict_window" and self.window_observations != self.lookback_periods + 1:
    raise ValueError("严格窗口观测数必须包含当前期和回看周期")
```

  `register` 在读取 `.name` 前先检查类型，避免对数字调用 `.strip()`；
  不实例化因子、不执行公式。现有 `names()` 排序逻辑保留。

```python
from inspect import isabstract
from quantlab.factors.base import BaseFactor, FactorDefinition

def register(self, factor_type: type[BaseFactor]) -> None:
    if not isinstance(factor_type, type) or not issubclass(factor_type, BaseFactor):
        raise ValueError("注册项必须是因子类型")
    if isabstract(factor_type):
        raise ValueError("不能注册抽象因子类型")
    name = getattr(factor_type, "name", None)
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("因子名称必须为非空且无两端空白的字符串")
    definition = getattr(factor_type, "definition", None)
    if not isinstance(definition, FactorDefinition) or definition.name != name:
        raise ValueError("因子定义或名称不一致")
    if name in self._factor_types:
        raise ValueError(f"因子已注册：{name}")
    self._factor_types[name] = factor_type
```

  `get` 对非字符串、空白名字显式抛中文 `ValueError`，有效但未知名字仍抛
  `KeyError`。包导出只增加两个结果类型，删除“尚无实现”的占位描述中
  与本任务不符部分。补齐坏查找参数的具体测试：

```python
@pytest.mark.parametrize("name", [None, [], "", " "])
def test_registry_invalid_lookup_name_has_clear_error(name):
    with pytest.raises(ValueError, match="名称"):
        FactorRegistry().get(name)
```

- [x] **Step 4：绿灯与任务边界复核。**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_factor_contracts.py -q
git diff --check
```

预期全部接口测试通过。检查只改因子接口和新测试，没有引入具体数据/回测依赖。

- [x] **Step 5：提交这一小步并解释。**

```bash
git add -- src/quantlab/factors/base.py src/quantlab/factors/registry.py src/quantlab/factors/__init__.py tests/test_factor_contracts.py
git diff --cached --check
git diff --cached --stat
git commit -m "feat: define factor contracts and explicit registry"
```

向用户说明：这一提交提供统一说明和调用方式，尚不能计算真实动量；冻结
数据类不意味着内部 pandas 表格不可修改，后续服务会复制与核对结果。

## Task 2：实现 20 期动量及严格窗口

**作用：** 得到第一个可手算核对的因子，先在单 ETF 内验证，不接页面或排名。

**Files:**

- Create: `src/quantlab/factors/momentum.py`、`tests/test_momentum_factor.py`。
- Modify: `src/quantlab/factors/registry.py`，增加工厂函数。
- Modify: `src/quantlab/factors/__init__.py`，导出类和工厂。

**Interfaces:**

- Consumes: Task 1 的 `BaseFactor`、`FactorDefinition`、`FactorEvaluation`、`FactorRegistry`。
- Produces: `Momentum20Factor.evaluate(market_data: pd.DataFrame) -> FactorEvaluation`；继承 `compute`。
- Produces: `create_default_registry() -> FactorRegistry`，每次新建，仅注册 `Momentum20Factor`。
- 私有 `_validate_input(market_data: pd.DataFrame) -> None` 不解析或修补输入，不修改调用方。
- `values` 名称固定为 `momentum_20`、dtype 为 `float64`；`status` 名称为 `status`，使用 pandas `string` dtype。

- [x] **Step 1：写公式、窗口和类型测试。** 以下代码同属新测试文件；
  使用合成工作日仅为了有自然日间隔，不把它称为官方交易日历。

```python
from dataclasses import asdict
import numpy as np
import pandas as pd
import pytest
from quantlab.factors import Momentum20Factor, create_default_registry

def prices(n=45):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-02", periods=n, freq="B").astype("datetime64[ns]"),
        "symbol": "510300", "close": 100.0 + np.arange(n) * 0.5,
        "has_observation": True,
    }, index=pd.Index(np.arange(n) * 3 + 7, name="source_row"))

def test_hand_calculation_metadata_and_default_registry():
    factor = Momentum20Factor()
    frame = prices(22)
    out = factor.evaluate(frame)
    assert out.values.iloc[:20].isna().all()
    assert out.status.iloc[:20].eq("warmup").all()
    assert out.values.iloc[20] == pytest.approx(0.10)
    assert out.values.iloc[21] == pytest.approx(110.5 / 100.5 - 1)
    assert out.status.iloc[20:].eq("valid").all()
    assert out.values.index.identical(frame.index)
    assert out.status.index.identical(frame.index)
    pd.testing.assert_series_equal(factor.compute(frame), out.values)
    assert asdict(factor.definition) == {
        "name": "momentum_20", "description": "20 期价格动量",
        "formula": "close[t] / close[t-20] - 1", "version": "1.0",
        "required_fields": ("close", "has_observation"),
        "lookback_periods": 20, "window_observations": 21,
        "direction": "higher_is_stronger", "missing_policy": "strict_window",
        "availability": "after_close",
    }
    left, right = create_default_registry(), create_default_registry()
    assert left is not right
    assert left.names() == right.names() == ("momentum_20",)
    assert left.get("momentum_20") is Momentum20Factor

@pytest.mark.parametrize("change", [0.0, -0.5])
def test_constant_and_falling_prices(change):
    frame = prices(21)
    frame["close"] = 100.0 + np.arange(21) * change
    actual = Momentum20Factor().compute(frame)
    assert actual.iloc[20] == pytest.approx(change * 20 / 100)

@pytest.mark.parametrize("gap", [0, 5, 20])
def test_full_window_missing_and_recovery(gap):
    frame = prices()
    row = frame.index[gap]
    frame.loc[row, ["close", "has_observation"]] = [np.nan, False]
    out = Momentum20Factor().evaluate(frame)
    assert out.status.iloc[:20].eq("warmup").all()
    assert out.status.iloc[20:gap + 21].eq("missing_window").all()
    assert out.values.iloc[20:gap + 21].isna().all()
    assert out.status.iloc[gap + 21] == "valid"

def test_short_all_missing_and_zero_volume():
    short = Momentum20Factor().evaluate(prices(20))
    assert short.values.isna().all() and short.status.eq("warmup").all()
    absent = prices().assign(close=np.nan, has_observation=False)
    out = Momentum20Factor().evaluate(absent)
    assert out.values.isna().all()
    assert out.status.iloc[20:].eq("missing_window").all()
    pd.testing.assert_series_equal(
        Momentum20Factor().compute(prices().assign(volume=0.0)),
        Momentum20Factor().compute(prices()),
    )

@pytest.mark.parametrize(("first", "last"), [
    (1e-308, 1e308), (1e308, 1e-308), (1e20, 1.0),
])
def test_unrepresentable_ratio_is_not_a_return(first, last):
    frame = prices(21)
    frame.loc[frame.index[[0, 20]], "close"] = [first, last]
    out = Momentum20Factor().evaluate(frame)
    assert out.status.iloc[20] == "numeric_range"
    assert pd.isna(out.values.iloc[20])
    assert not np.isinf(out.values.to_numpy()).any()

def test_only_past_rows_are_used_and_input_is_untouched():
    frame = prices()
    before = frame.copy(deep=True)
    early = Momentum20Factor().evaluate(frame)
    changed = frame.copy(deep=True)
    changed.loc[changed.index[30:], "close"] *= 2
    later = Momentum20Factor().evaluate(changed)
    pd.testing.assert_series_equal(early.values.iloc[:30], later.values.iloc[:30])
    pd.testing.assert_series_equal(early.status.iloc[:30], later.status.iloc[:30])
    early.values.iloc[-1] = 999.0
    pd.testing.assert_frame_equal(frame, before)

@pytest.mark.parametrize("mutate", [
    lambda f: f.iloc[:0],
    lambda f: f.drop(columns="close"),
    lambda f: pd.concat([f, f[["close"]]], axis=1),
    lambda f: f.set_axis([0] * len(f)),
    lambda f: f.iloc[::-1],
    lambda f: f.assign(date=f.date.iloc[0]),
    lambda f: f.assign(date=f.date.astype(str)),
    lambda f: f.assign(date=f.date.astype("datetime64[us]")),
    lambda f: f.assign(date=f.date.dt.tz_localize("UTC")),
    lambda f: f.assign(date=f.date + pd.Timedelta(hours=1)),
    lambda f: f.assign(date=f.date.mask(f.index == f.index[0])),
    lambda f: f.assign(symbol=510300),
    lambda f: f.assign(symbol="51030"),
    lambda f: f.assign(symbol=["159915"] + ["510300"] * (len(f) - 1)),
    lambda f: f.assign(symbol=None),
    lambda f: f.assign(close=f.close.astype("float32")),
    lambda f: f.assign(close=f.close.astype("Float64")),
    lambda f: f.assign(close=f.close.astype(str)),
    lambda f: f.assign(close=True),
    lambda f: f.assign(close=0.0),
    lambda f: f.assign(close=-1.0),
    lambda f: f.assign(close=np.inf),
    lambda f: f.assign(close=np.nan),
    lambda f: f.assign(has_observation=1),
    lambda f: f.assign(has_observation=False),
    lambda f: f.assign(has_observation=pd.Series(pd.NA, index=f.index, dtype="boolean")),
])
def test_invalid_input_is_not_silently_normalized(mutate):
    with pytest.raises(ValueError):
        Momentum20Factor().evaluate(mutate(prices()))

def test_non_dataframe_rejected_and_nullable_bools_without_na_allowed():
    with pytest.raises(ValueError):
        Momentum20Factor().evaluate(None)
    frame = prices()
    expected = Momentum20Factor().compute(frame)
    frame["has_observation"] = frame.has_observation.astype("boolean")
    pd.testing.assert_series_equal(Momentum20Factor().compute(frame), expected)
```

- [x] **Step 2：确认红灯由缺少因子实现引起。**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_momentum_factor.py -q
```

首次预期 `Momentum20Factor` 导入失败；Task 1 测试此时仍须通过。

- [x] **Step 3：实现输入验证、公式和显式默认注册。** 输入顺序不能重排。
  先检查非空 DataFrame、唯一列名、四个必需列、唯一行索引；再检查单一
  六位字符串代码、日期、数值和观测标记。日期只验证，不调用字符串解析。
  关键验证分支如下，其余检查用上一句对应条件逐项抛中文 `ValueError`：

```python
import re
import numpy as np
import pandas as pd

if not market_data.symbol.map(
    lambda value: isinstance(value, str) and re.fullmatch(r"[0-9]{6}", value) is not None
).all() or market_data.symbol.nunique() != 1:
    raise ValueError("因子输入必须是一只六位字符串代码的 ETF")
dates = market_data["date"]
if dates.dtype != np.dtype("datetime64[ns]"):
    raise ValueError("因子日期必须是无时区的 datetime64[ns] 日线")
if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
    raise ValueError("因子日期必须非空、唯一且升序")
if not dates.eq(dates.dt.normalize()).all():
    raise ValueError("因子日期不能包含盘中时间")
observed = market_data["has_observation"]
if not pd.api.types.is_bool_dtype(observed.dtype) or observed.isna().any():
    raise ValueError("观测标记必须为无空值布尔列")
close = market_data["close"]
if close.dtype != np.dtype("float64"):
    raise ValueError("收盘价必须保持规范 float64 类型")
present = close.loc[observed].to_numpy(dtype="float64")
if not np.isfinite(present).all() or (present <= 0).any():
    raise ValueError("观测行收盘价必须为有限正数")
if not close.loc[~observed].isna().all():
    raise ValueError("缺失行收盘价必须为 NaN，不能填充")
```

  `Momentum20Factor` 的定义字段严格对应 Step 1 的完整字典。`evaluate`
  调用验证后用位置生成窗口；只为完整窗口运算，局部屏蔽预期浮点警告，
  不全局修改 NumPy 错误模式：

```python
close = market_data.close.to_numpy(dtype="float64", copy=True)
count = len(close)
values = np.full(count, np.nan, dtype="float64")
status = np.full(count, "warmup", dtype=object)
status[20:] = "missing_window"
complete = market_data.has_observation.astype("int64").rolling(21, min_periods=21).sum().eq(21)
positions = np.flatnonzero(complete.to_numpy())
with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
    candidate = close[positions] / close[positions - 20] - 1.0
representable = np.isfinite(candidate) & (candidate > -1.0)
status[positions] = "numeric_range"
status[positions[representable]] = "valid"
values[positions[representable]] = candidate[representable]
return FactorEvaluation(
    pd.Series(values, index=market_data.index.copy(), name="momentum_20", dtype="float64"),
    pd.Series(status, index=market_data.index.copy(), name="status", dtype="string"),
)
```

  `registry.py` 仅增加显式工厂，局部导入具体类避免基础注册表循环依赖：

```python
def create_default_registry() -> FactorRegistry:
    from quantlab.factors.momentum import Momentum20Factor
    registry = FactorRegistry()
    registry.register(Momentum20Factor)
    return registry
```

- [x] **Step 4：合并测试绿灯。**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_factor_contracts.py tests/test_momentum_factor.py -q
git diff --check
```

核对前 20 个位置为空，第 21 个位置是 10%，间隔跨周末不改变周期计数；
不增加把结果整体滞后一日的代码，因为当前输出是收盘后可用的描述值。

- [x] **Step 5：提交并解释公式。**

```bash
git add -- src/quantlab/factors/momentum.py src/quantlab/factors/registry.py src/quantlab/factors/__init__.py tests/test_momentum_factor.py
git diff --cached --check
git diff --cached --stat
git commit -m "feat: calculate strict-window 20-period price momentum"
```

向用户解释：20 期动量是“现在的价格相对 20 个日期轴位置前涨跌多少”，
不是获利预测；首 20 条热身留空，中间缺数据也不能跳过。

## Task 3：连接研究快照、追溯和真实端到端验收

**作用：** 把单 ETF 公式放进现有多 ETF 研究流程，保留数据身份，并证明重读
快照后仍能独立计算。本步没有新页面，不作浏览器验收或因子有效性评价。

**Files:**

- Create: `src/quantlab/factors/research.py`、`tests/test_factor_research.py`、`docs/factors.md`。
- Modify: `src/quantlab/factors/__init__.py`，增加研究入口和结果导出。
- Read only: `src/quantlab/data/research_sets.py`、`research_storage.py`、`calendar.py`、`alignment.py`。

**Interfaces:**

- Consumes: `validate_research_preview(preview: ResearchSetPreview) -> None`，只校验内存，不回读来源。
- Consumes: `create_default_registry()`、`FactorRegistry.get()` 及 `BaseFactor.evaluate()`。
- Produces: `run_research_factor(preview, factor_name="momentum_20", *, registry=None)`。
- Produces: `FactorRunResult`；`data` 固定五列，日期/代码保持输入网格顺序。
- 私有 `_validate_evaluation(evaluation: FactorEvaluation, expected_index: pd.Index) -> None` 验证插件输出；不修补、不重排。

```python
from dataclasses import dataclass
from datetime import date
import pandas as pd
from quantlab.data.calendar import CalendarAxis
from quantlab.data.research_sets import ResearchSetPreview
from quantlab.factors.base import FactorDefinition, FactorEvaluation
from quantlab.factors.registry import FactorRegistry

@dataclass(frozen=True)
class FactorRunResult:
    data: pd.DataFrame
    definition: FactorDefinition
    input_preview_sha256: str
    alignment_content_sha256: str
    start: date
    end: date
    calendar: CalendarAxis
    adjustment: str
    sources: tuple[dict[str, str], ...]
    warnings: tuple[str, ...]
```

- [ ] **Step 1：建立仅本测试文件使用的真实导入夹具及核心红灯测试。**
  不改共享 `conftest.py`。每次在 `tmp_path` 的独立子目录创建合成版本；
  该测试工厂允许造缺失、日期轴差异及后续价格变化，不 mock 数据层。

```python
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from quantlab.data.calendar import parse_calendar_csv
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.research_sets import DatasetSelection, build_research_set, SOURCE_FIELDS
from quantlab.data.research_storage import save_research_set, load_research_set
from quantlab.factors import (
    BaseFactor, FactorEvaluation, FactorRegistry, Momentum20Factor, run_research_factor,
)

@pytest.fixture
def make_preview(tmp_path):
    sequence = count()

    def build(*, gaps=None, outside=None, supplied=False, adjustment="none",
              bump_after=None, start_position=0, end_position=44):
        source_root = tmp_path / f"sources_{next(sequence)}"
        source_root.mkdir()
        raw, processed = source_root / "raw", source_root / "processed"
        dates = pd.date_range("2025-01-02", periods=45, freq="B").astype("datetime64[ns]")
        for symbol, values in (("510300", 100.0 + np.arange(45) * 0.5),
                               ("159915", 200.0 - np.arange(45))):
            if bump_after is not None:
                values[bump_after:] *= 2
            asset_dates = dates + pd.Timedelta(days=365) if outside == symbol else dates
            frame = pd.DataFrame({
                "date": asset_dates, "symbol": symbol, "open": values,
                "high": values, "low": values, "close": values, "volume": 0.0,
            })
            frame = frame.drop(index=(gaps or {}).get(symbol, []))
            path = source_root / f"{symbol}.csv"
            frame.to_csv(path, index=False)
            import_market_csv(path, raw, processed,
                              imported_at=datetime(2025, 5, 1, tzinfo=timezone.utc))
        calendar = None
        if supplied:
            content = ("date\n" + "\n".join(day.date().isoformat() for day in dates) + "\n").encode()
            calendar = parse_calendar_csv(content, source="合成声明日历",
                                          coverage_start=dates[0].date(), coverage_end=dates[-1].date())
        scan = scan_market_catalog(processed, raw)
        assert not scan.issues and len(scan.records) == 2
        preview = build_research_set(
            tuple(DatasetSelection(record, adjustment) for record in scan.records),
            start=dates[start_position].date(), end=dates[end_position].date(), calendar=calendar,
        )
        return preview, source_root
    return build

def test_multi_etf_values_grid_and_provenance(make_preview):
    preview, _ = make_preview(supplied=True)
    before = deepcopy(preview)
    result = run_research_factor(preview)
    assert result.data.columns.tolist() == ["date", "symbol", "factor_value", "is_valid", "status"]
    pd.testing.assert_frame_equal(result.data[["date", "symbol"]], preview.alignment.data[["date", "symbol"]])
    up = result.data.loc[result.data.symbol.eq("510300")]
    down = result.data.loc[result.data.symbol.eq("159915")]
    assert up.factor_value.iloc[20] == pytest.approx(0.10)
    assert down.factor_value.iloc[20] == pytest.approx(-0.10)
    pd.testing.assert_series_equal(result.data.is_valid,
                                   result.data.status.eq("valid").astype(bool).rename("is_valid"))
    assert result.input_preview_sha256 == preview.preview_sha256
    assert result.alignment_content_sha256 == preview.alignment.content_sha256
    assert asdict(result.calendar) == asdict(preview.alignment.calendar)
    assert (result.start, result.end, result.adjustment) == (preview.start, preview.end, "none")
    assert result.definition == Momentum20Factor.definition
    assert set(result.sources[0]) == set(SOURCE_FIELDS) - {"processed_path"}
    assert result.sources == tuple({k: v for k, v in s.items() if k != "processed_path"} for s in preview.sources)
    assert all(w in result.warnings for w in preview.warnings)
    assert any("总回报" in w for w in result.warnings)
    assert any("收盘" in w for w in result.warnings)
    pd.testing.assert_frame_equal(preview.alignment.data, before.alignment.data)
    pd.testing.assert_frame_equal(preview.coverage, before.coverage)
    assert preview.sources == before.sources
    result.data.loc[0, "factor_value"] = 999.0
    result.sources[0]["provider"] = "changed-output"
    assert preview.sources == before.sources
    pd.testing.assert_frame_equal(preview.alignment.data, before.alignment.data)

@pytest.mark.parametrize("labels", ["duplicate", "noncontiguous"])
def test_research_index_labels_are_not_data_identity(make_preview, labels):
    preview, _ = make_preview()
    expected = run_research_factor(preview)
    size = len(preview.alignment.data)
    new_index = pd.Index([0] * size if labels == "duplicate" else np.arange(size) * 3, name="date")
    preview.alignment.data.index = new_index
    actual = run_research_factor(preview)
    pd.testing.assert_frame_equal(actual.data, expected.data)
    assert actual.input_preview_sha256 == expected.input_preview_sha256
    assert preview.alignment.data.index.identical(new_index)

def test_all_missing_asset_stays_and_short_snapshot_does_not_reload_history(make_preview):
    preview, _ = make_preview(outside="159915")
    result = run_research_factor(preview)
    absent = result.data.loc[result.data.symbol.eq("159915")]
    assert len(absent) == 45 and absent.factor_value.isna().all()
    assert absent.status.iloc[:20].eq("warmup").all()
    assert absent.status.iloc[20:].eq("missing_window").all()
    short, root = make_preview(start_position=25)
    root.rename(root.with_name(root.name + "_moved"))
    unavailable = run_research_factor(short)
    assert len(unavailable.data) == 40
    assert unavailable.data.status.eq("warmup").all()
    assert not unavailable.data.is_valid.any()
    assert any("没有可用" in w for w in unavailable.warnings)

def test_union_and_supplied_calendar_are_not_interchangeable(make_preview):
    gaps = {"510300": [5], "159915": [5]}
    union, _ = make_preview(gaps=gaps)
    supplied, _ = make_preview(gaps=gaps, supplied=True)
    union_out, supplied_out = run_research_factor(union), run_research_factor(supplied)
    day = pd.Timestamp(supplied.alignment.calendar.dates[21])
    a = union_out.data.loc[lambda f: f.date.eq(day) & f.symbol.eq("510300")].iloc[0]
    b = supplied_out.data.loc[lambda f: f.date.eq(day) & f.symbol.eq("510300")].iloc[0]
    assert a.status == "valid" and a.factor_value == pytest.approx(0.105)
    assert b.status == "missing_window" and pd.isna(b.factor_value)
    assert union_out.input_preview_sha256 != supplied_out.input_preview_sha256
    assert supplied_out.calendar.file_sha256 == supplied.alignment.calendar.file_sha256
    assert supplied_out.calendar.mode == "supplied" and union_out.calendar.mode == "observed_union"

@pytest.mark.parametrize("adjustment", ["qfq", "hfq"])
def test_adjustment_declaration_retained_not_recomputed(make_preview, adjustment):
    preview, _ = make_preview(adjustment=adjustment)
    result = run_research_factor(preview)
    assert result.adjustment == adjustment
    assert all(s["adjustment_basis"] == "user_declared" for s in result.sources)
    assert all(w in result.warnings for w in preview.warnings)
    assert any("当时可得" in w for w in result.warnings)
    assert any("同一复权标签" in w for w in result.warnings)

def test_later_price_change_cannot_change_earlier_results(make_preview):
    original, _ = make_preview()
    changed, _ = make_preview(bump_after=30)
    left, right = run_research_factor(original), run_research_factor(changed)
    cutoff = pd.Timestamp(original.alignment.calendar.dates[30])
    pd.testing.assert_frame_equal(left.data.loc[left.data.date.lt(cutoff)], right.data.loc[right.data.date.lt(cutoff)])
    assert left.input_preview_sha256 != right.input_preview_sha256

@pytest.mark.parametrize("mutation", ["price", "source", "extra_source", "calendar", "hash", "coverage", "warnings"])
def test_mutated_preview_is_rejected_not_repaired(research_preview, mutation):
    preview = deepcopy(research_preview)
    if mutation == "price":
        preview.alignment.data.loc[preview.alignment.data.has_observation, "close"] = 3.15
    elif mutation == "source":
        preview.sources[0]["version_id"] = "changed-version"
    elif mutation == "extra_source":
        preview.sources[0]["secret"] = "synthetic-only"
    elif mutation == "calendar":
        preview = replace(preview, alignment=replace(preview.alignment,
                          calendar=replace(preview.alignment.calendar, source="changed")))
    elif mutation == "hash":
        preview = replace(preview, preview_sha256="0" * 64)
    elif mutation == "coverage":
        preview.coverage.loc[0, "missing_count"] = 123
    else:
        preview = replace(preview, warnings=())
    with pytest.raises(ValueError):
        run_research_factor(preview)

def test_invalid_entry_arguments(research_preview):
    with pytest.raises(ValueError, match="预览|研究"):
        run_research_factor(None)
    with pytest.raises(KeyError, match="未注册"):
        run_research_factor(research_preview, "unknown")
    with pytest.raises(ValueError, match="名称"):
        run_research_factor(research_preview, [])
    with pytest.raises(ValueError, match="注册表"):
        run_research_factor(research_preview, registry={})
```

- [ ] **Step 2：先运行新增文件确认红灯。**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_factor_research.py -q
```

预期尚未导出的 `run_research_factor` 导致失败。再写入下一步的坏因子及
保存链路测试；不要先实现服务再补上所有测试。

- [ ] **Step 3：补上不合约结果、计算输入隔离及保存重读测试。** 以下继续
  追加到同一测试文件，使用上文已定义的 `make_preview` 和 imports。

```python
@pytest.mark.parametrize("damage", [
    "wrong_type", "short", "reversed", "index_name", "int_values", "infinity",
    "bad_state", "na_state", "numeric_state", "valid_nan", "invalid_number",
])
def test_bad_factor_output_fails_instead_of_silent_alignment(make_preview, damage):
    class BrokenFactor(Momentum20Factor):
        name = "broken"
        definition = replace(Momentum20Factor.definition, name=name)

        def evaluate(self, frame):
            out = super().evaluate(frame)
            values, status = out.values.copy(), out.status.copy()
            if damage == "wrong_type":
                return values
            if damage == "short":
                values = values.iloc[:-1]
            elif damage == "reversed":
                status = status.iloc[::-1]
            elif damage == "index_name":
                values.index = values.index.rename("wrong")
            elif damage == "int_values":
                values = values.fillna(0).astype("int64")
            elif damage == "infinity":
                values.iloc[-1] = np.inf
            elif damage == "bad_state":
                status.iloc[0] = "unknown"
            elif damage == "na_state":
                status.iloc[0] = pd.NA
            elif damage == "numeric_state":
                status = pd.Series(1, index=frame.index)
            elif damage == "valid_nan":
                status.iloc[0] = "valid"
            elif damage == "invalid_number":
                values.iloc[0] = 0.0
            return FactorEvaluation(values, status)

    registry = FactorRegistry()
    registry.register(BrokenFactor)
    preview, _ = make_preview()
    with pytest.raises(ValueError, match="因子.*输出"):
        run_research_factor(preview, "broken", registry=registry)

def test_factor_cannot_rewrite_merge_keys_or_caller_data(make_preview):
    class MutatingFactor(Momentum20Factor):
        name = "mutating"
        definition = replace(Momentum20Factor.definition, name=name)

        def evaluate(self, frame):
            out = super().evaluate(frame)
            frame["date"] = frame.date + pd.Timedelta(days=100)
            frame["symbol"] = "000000"
            frame["close"] = 1.0
            return out

    preview, _ = make_preview()
    before = preview.alignment.data.copy(deep=True)
    registry = FactorRegistry()
    registry.register(MutatingFactor)
    expected = run_research_factor(preview)
    result = run_research_factor(preview, "mutating", registry=registry)
    pd.testing.assert_frame_equal(result.data, expected.data)
    pd.testing.assert_frame_equal(preview.alignment.data, before)

def test_saved_snapshot_calculates_without_original_files(make_preview, tmp_path, monkeypatch):
    preview, sources = make_preview(supplied=True)
    root = tmp_path / "sets"
    saved = save_research_set(preview, root, name="合成动量验收")
    saved_bytes = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    sources.rename(sources.with_name(sources.name + "_moved"))
    loaded = load_research_set(root, saved.set_id)
    assert all(not Path(s["processed_path"]).exists() for s in loaded.preview.sources)
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "synthetic-secret-not-for-results")

    def no_file_access(*args, **kwargs):
        raise AssertionError("因子计算不应读写文件")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", no_file_access)
        patch.setattr("builtins.open", no_file_access)
        patch.setattr(pd, "read_parquet", no_file_access)
        result = run_research_factor(loaded.preview)
    assert result.data.loc[result.data.symbol.eq("510300"), "factor_value"].iloc[20] == pytest.approx(0.10)
    assert result.input_preview_sha256 == preview.preview_sha256
    assert result.definition.version == "1.0"
    assert result.calendar.file_sha256 == preview.alignment.calendar.file_sha256
    assert "synthetic-secret-not-for-results" not in repr(result)
    assert all("processed_path" not in source for source in result.sources)
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == saved_bytes
```

  仅移动本次测试生成的目录，不删除用户文件。不将两端版本/hash 伪造为
  “相同”来强行过测试；实际使用真实导入、保存和载入，只有输入/输出契约
  测试使用故意损坏的因子替身。

- [ ] **Step 4：实现纯内存研究编排与输出校验。** 入口先检查参数类型，
  `deepcopy(preview)` 后调用现有校验；严禁调用 `research_preview_sha256`
  重算并覆盖旧指纹以修复输入。运行步骤的核心代码：

```python
from copy import deepcopy
import numpy as np
from quantlab.data.research_sets import SOURCE_FIELDS, validate_research_preview
from quantlab.factors.registry import create_default_registry

def run_research_factor(
    preview: ResearchSetPreview, factor_name: str = "momentum_20", *,
    registry: FactorRegistry | None = None,
) -> FactorRunResult:
    if not isinstance(preview, ResearchSetPreview):
        raise ValueError("研究输入必须是 ResearchSetPreview")
    if registry is not None and not isinstance(registry, FactorRegistry):
        raise ValueError("因子注册表类型无效")
    snapshot = deepcopy(preview)
    validate_research_preview(snapshot)
    active = create_default_registry() if registry is None else registry
    factor_type = active.get(factor_name)
    definition = deepcopy(factor_type.definition)
    factor = factor_type()
    frame = snapshot.alignment.data.reset_index(drop=True)
    keys = frame[["date", "symbol"]].copy(deep=True)
    parts = []
    for symbol in tuple(source["symbol"] for source in snapshot.sources):
        asset = frame.loc[frame.symbol.eq(symbol)].copy(deep=True)
        asset_keys = asset[["date", "symbol"]].copy(deep=True)
        evaluation = factor.evaluate(asset.copy(deep=True))
        _validate_evaluation(evaluation, asset.index)
        asset_keys["factor_value"] = evaluation.values.copy(deep=True)
        asset_keys["status"] = evaluation.status.copy(deep=True)
        parts.append(asset_keys)
    calculated = pd.concat(parts, ignore_index=True)
    combined = keys.merge(calculated, on=["date", "symbol"], how="left",
                          sort=False, validate="one_to_one")
    if not combined[["date", "symbol"]].equals(keys) or combined.status.isna().any():
        raise ValueError("因子输出无法与原研究网格逐行对应")
    combined["is_valid"] = combined.status.eq("valid").astype(bool)
    data = combined[["date", "symbol", "factor_value", "is_valid", "status"]].copy(deep=True)
    sources = tuple({key: source[key] for key in SOURCE_FIELDS if key != "processed_path"}
                    for source in snapshot.sources)
    warnings = _factor_warnings(snapshot, bool(data.is_valid.any()))
    return FactorRunResult(
        data=data, definition=definition,
        input_preview_sha256=snapshot.preview_sha256,
        alignment_content_sha256=snapshot.alignment.content_sha256,
        start=snapshot.start, end=snapshot.end, calendar=deepcopy(snapshot.alignment.calendar),
        adjustment=snapshot.adjustment, sources=deepcopy(sources), warnings=warnings,
    )
```

  将上述未展开的两个私有函数也放在 `research.py`。`_validate_evaluation`
  先验证是 `FactorEvaluation` 且两个字段是 Series，随后：

```python
def _validate_evaluation(evaluation: FactorEvaluation, expected_index: pd.Index) -> None:
    if not isinstance(evaluation, FactorEvaluation):
        raise ValueError("因子输出必须包含数值和状态")
    values, status = evaluation.values, evaluation.status
    if not isinstance(values, pd.Series) or not isinstance(status, pd.Series):
        raise ValueError("因子输出数值与状态必须为 Series")
    if not values.index.identical(expected_index) or not status.index.identical(expected_index):
        raise ValueError("因子输出索引、顺序或长度与输入不一致")
    if values.dtype != np.dtype("float64"):
        raise ValueError("因子输出数值必须为 float64")
    allowed = {"warmup", "missing_window", "numeric_range", "valid"}
    if not isinstance(status.dtype, pd.StringDtype) or status.isna().any() or not status.isin(allowed).all():
        raise ValueError("因子输出状态无效")
    valid = status.eq("valid")
    if not np.isfinite(values.loc[valid].to_numpy(dtype="float64")).all():
        raise ValueError("因子输出有效行必须是有限数值")
    if not values.loc[~valid].isna().all():
        raise ValueError("因子输出不可用行必须为 NaN")
```

  `_factor_warnings(preview: ResearchSetPreview, has_valid: bool) -> tuple[str, ...]`
  继承全部原警告，再用下列固定文案。不给出预测、盈利或可成交结论：

```python
def _factor_warnings(preview: ResearchSetPreview, has_valid: bool) -> tuple[str, ...]:
    messages = list(preview.warnings)
    messages.extend([
        "20 期按研究日期轴位置计算，不代表 20 个自然日；并集可能漏掉全体缺失日。",
        "前 20 个日期位置为热身期；窗口内任一缺失都留空，不自动读取区间外历史。",
        "因子含当日收盘价，只能在收盘数据可用后计算，不是交易指令。",
        "未使用未来行不等于输入在历史当时可得；本结果是固定数据版本上的回顾性研究。",
        "同一复权标签不保证来源算法相同；口径声明不等于认证，未在此进行复权转换。",
    ])
    if preview.adjustment == "none":
        messages.append("未复权价格可能受分红、拆并份额影响，价格动量不能当作总回报。")
    if not has_valid:
        messages.append("本研究区间没有可用因子值，请核对历史长度和窗口内缺失情况。")
    return tuple(dict.fromkeys(messages))
```

  公共导出增加 `FactorRunResult`、`run_research_factor`。禁止调用行情
  Provider、读取环境变量、自动保存因子表或改动 `aligned.parquet`。

- [ ] **Step 5：运行专项测试，失败时定位输入或契约而不是放宽规则。**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_factor_contracts.py tests/test_momentum_factor.py tests/test_factor_research.py -q -rs
PYTHONPATH=src .venv/bin/python -m pytest tests/test_research_sets.py tests/test_research_storage.py tests/test_alignment.py tests/test_calendar.py -q -rs
```

  预期全部通过。特别核对：全缺失资产没有消失；默认表只含一个因子；
  来源中的版本、三个指纹及复权依据保留；日期轴全部字段一致。

- [ ] **Step 6：编写学习说明和运行用法。** `docs/factors.md` 按以下顺序：
  模块职责（接口/公式/研究服务）→ 100 到 110 的手算 → 四种状态表 →
  从现有快照调用 → 数据/公式身份 → 限制 → 本次实际验收记录。
  使用下面函数示例，不硬编码真实用户的路径或虚构集合 ID：

```python
from pathlib import Path
from quantlab.data.research_storage import load_research_set
from quantlab.factors import FactorRunResult, run_research_factor

def calculate_saved_momentum(research_dir: Path, set_id: str) -> FactorRunResult:
    loaded = load_research_set(research_dir, set_id)
    return run_research_factor(loaded.preview, "momentum_20")
```

  `research_dir` 是既有研究集合目录，`set_id` 来自已有保存记录；说明当前
  没有界面按钮。逐行解释：载入时核验快照 → 计算并返回副本 → `result.data`
  供后续分析，`definition/sources/calendar` 供追溯。明确保留 20 期热身、
  严格窗口、不能推断真实交易能力、无盈利证明、无历史当时可得认证。
  不把缺失状态表写成数据源故障诊断；它只描述当前输入为什么算不出值。

- [ ] **Step 7：完整回归、依赖和独立性检查，填写实际结果。**

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -rs
.venv/bin/python -m pip check
git diff --check
rg -n 'quantlab\.factors|from .*factors' src/quantlab/backtest src/quantlab/analytics src/quantlab/strategies
git status --short
```

  最后一项 `rg` 无匹配时退出码 1 为预期；若某路径不存在，先用
  `rg --files src/quantlab` 找出真实第一阶段模块，不能把路径错误当作
  独立性验证成功。验收记录写实际命令、通过/跳过原因和范围；任何已有
  未提交改动导致的基线差异须说明，不替无关功能背书。不声称做了 UI
  验收、真实市场有效性检验或外部 Provider 验证。

- [ ] **Step 8：提交完整可用链路并进行所选执行流程要求的独立审查。**

```bash
git add -- src/quantlab/factors/research.py src/quantlab/factors/__init__.py tests/test_factor_research.py docs/factors.md
git diff --cached --check
git diff --cached --stat
git commit -m "feat: run traceable factors on validated research snapshots"
```

  审查范围只包含本计划产生的提交；不把第一阶段/iFinD 的脏改动纳入。
  若审查发现问题，先写失败测试再修复，重跑受影响测试及最终完整回归。
  保留当前分支和目录，不自动合并、推送或部署。

## 设计覆盖及交付检查

| 设计要求 | 对应任务与证据 |
|---|---|
| 统一说明、版本、旧 compute 调用方式、显式注册 | Task 1；Task 2 元数据和工厂测试 |
| 规范输入、21 条完整窗口、四种状态、极端数值 | Task 2 参数化边界和手算测试 |
| 逐 ETF 独立、按原网格输出、输入不变 | Task 3 多资产、索引及恶意修改替身测试 |
| 口径/来源/完整日期轴/双指纹、缺失警告 | Task 3 来源、日历差异及声明测试 |
| 未来价格隔离、区间外历史不读取 | Task 2 时间隔离；Task 3 重建预览和短区间测试 |
| 快照不变、来源移走仍可重读后计算 | Task 3 真实保存重读及字节比对 |
| 无新依赖/界面/排名/交易，原功能保持 | Task 3 完整回归、依赖和引用检查 |

## 执行记录（2026-09-23）

用户已审阅计划并选择推荐方式：当前会话内逐项实现，最后统一独立审查。
已按用户逐步学习要求先后交付 Task 1、Task 2；Task 3 尚未开始，已有
单 ETF 动量计算，但尚未连接研究快照、追溯输出或因子界面。整份计划的
最终独立审查尚未进行。

- Task 1：统一说明、数值/状态接口和显式注册已完成；现有 `compute` 调用形式保留。
- Task 1 红灯：新增测试因 `FactorDefinition` 尚不存在而失败，与计划预期一致。
- Task 1 验证：84 项通过；全项目回归：706 项通过、2 项可选外部接口测试跳过。
- Task 1 对照基线：622 项通过、相同的 2 项跳过；依赖一致性检查通过。
- Task 2：实现 `Momentum20Factor` 和独立默认注册表；严格 21 条窗口、四种逐行状态、索引保留和输入不修改。
- Task 2 红灯：新增测试因 `Momentum20Factor` 尚不存在而失败；同时原有 84 项因子接口测试通过。
- Task 2 验证：新增 45 项测试通过，因子专项共 129 项通过；全项目 751 项通过、2 项外部接口测试跳过；依赖一致性检查通过。
- Task 2 对照基线：706 项通过、相同的 2 项跳过；没有运行外部数据服务或读取私人行情。
- Task 2 手算：20 期前 100、当前 110 且窗口完整时为 10%；前 20 个位置留空，窗口中间缺失亦留空。
- 仅提交因子基础代码、新测试及本计划进度，未修改第一阶段/iFinD 的既有脏改动。
- 继续使用原目录与虚拟环境；不重新安装依赖、不新建工作树、不推送或部署。
