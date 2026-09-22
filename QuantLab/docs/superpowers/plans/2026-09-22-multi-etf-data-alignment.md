# 多 ETF 数据管理与交易日对齐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已导入的多只沪深 ETF 整理为版本固定、缺失可见、可保存重读的日线研究集合。

**Architecture:** 复用现有单 ETF 导入和版本文件，将目录读取提取到不依赖 Streamlit 的数据层。新增纯函数日历及对齐模块，再由研究集合服务连接载入、校验和持久化，最后接入独立页面。单资产回测和具体因子不依赖本次新功能。

**Tech Stack:** 现有 Python 3.12、pandas、NumPy、PyArrow、pytest、Streamlit AppTest；不增加依赖，不联网。

**Spec:** `docs/superpowers/specs/2026-09-22-multi-etf-data-alignment-design.md`（用户已确认；执行者先完整阅读该文件）。

## Global Constraints

- “本次只做第二阶段的数据基础，不一次实现整个第二阶段。”
- “本次不支持境外交易所市场、跨时区或多币种对齐。”
- “一个集合至少包含两个不同 ETF 代码，每个代码只选择一个版本。”
- “集合内所有 ETF 必须采用相同的 `none/qfq/hfq` 价格口径。”
- “不做前向填充、后向填充、插值或自动补零。”
- “不输出 `tradable` 等确定可交易状态”。
- “原始有效数值保持不变，统一数值类型，不改变原始 DataFrame。”
- “规则版本使用 `multi-etf-alignment-v1`。”
- “不扫描或保存环境变量、凭证、个人交易。”
- 所有测试仅使用合成数据、现有夹具与临时目录；不读取私人行情或钥匙串。
- 保护现有未提交改动，尤其第一阶段收尾和 iFinD 接入；不使用全目录暂存。
- 按用户要求逐步构建，完成每项任务后用中文解释其作用和验证结果。

## Review Focus

以下五类输入在任务中有对应的明确测试，审查时不能只看正常路径：

1. 版本元数据有重复、错配或越界路径：只读取当前数据根目录中的完整版本，不误选同名版本（Task 1）。
2. CSV 带 BOM、同一日期不同表示或盘中时间：保留合法 UTF-8 BOM 输入，标准化后查重，不截断时间来伪装日线（Task 2、3）。
3. 一只 ETF 在区间内完全没有数据、另一只有零成交量：分别呈现全缺失与真实观测，不把两者混同（Task 3）。
4. 保存前对象被改动、保存中断、文件被篡改：不发布伪造或半成品集合，重读时检查内容和语义（Task 4、5）。
5. 用户预览后更换版本或日历、公开会话重新运行页面：旧预览不能保存，私有目录不能泄漏（Task 6）。

---

## 执行环境与小步交付

项目目录为 `/Users/zhoujiayu/Documents/量化/QuantLab`，Git 根目录在其上一级。
下文源文件和测试路径均相对项目目录，命令在项目目录运行；Git 提交仅列出
本任务实际改动文件。阶段 1 最近一次验收为 256 passed、2 个联网测试跳过，
这是历史基线，不应替代执行前后的新测试。

当前阶段 1 收尾仍有未提交文件，`a95fe43` 只提交了本次设计文档。
执行前必须确认工作树状态并运行基线测试。若采用隔离工作树，不能仅从
HEAD 创建后忽略这些文件；先按工作树技能确认如何保留完整当前基线。
不得自动提交、丢弃或搬走不属于当前任务的改动。

```bash
git status --short
.venv/bin/python -m pytest -q -rs
```

出现新的基线失败时先说明原因，不能为进入新功能而删除旧测试。
业务代码实现从 Task 1 开始；本计划文件中的代码是接口与测试指导，
不是已完成实现。每项任务内部按“红灯测试 → 最小实现 → 绿灯验证”推进，
不要一次生成所有产品代码。

## 文件职责

| 文件 | 职责 |
|---|---|
| `src/quantlab/data/catalog.py` | 完整版本目录扫描、选择身份核对、单次文件快照读取 |
| `src/quantlab/data/calendar.py` | 日线日期校验、CSV 日历输入和日期轴选择 |
| `src/quantlab/data/alignment.py` | 规范化、对齐、观测掩码、质量统计和确定性内容指纹 |
| `src/quantlab/data/research_sets.py` | 研究集合请求、输入口径确认与预览快照编排 |
| `src/quantlab/data/research_storage.py` | 保存、完成标记、校验重读与集合列表 |
| `src/quantlab/ui/multi_asset.py` | 页面请求指纹、友好错误与可保存预览状态 |
| `src/quantlab/ui/runtime.py` | 沿用本地/会话隔离，增加研究集合目录 |
| `pages/6_多ETF数据.py`、`app.py` | 新页面和导航入口 |
| `tests/conftest.py` | 仅测试使用的两 ETF 合成数据及导入夹具 |
| `docs/multi_etf_data.md` | 使用步骤、模块作用、缺失语义和限制 |

设计文档中的 `research_sets.py` 按职责拆为编排与存储两个小模块，公开
工作流不变。旧 `ui/market_data.py` 只替换目录扫描部分，不改 Provider、
友好导入或凭证相关逻辑。

## Task 1：将行情版本目录提取到数据层

**作用：** 让第二阶段可以可靠选择具体 ETF 和版本，同时保留第一阶段页面接口。

**Files:**

- Create: `src/quantlab/data/catalog.py`
- Modify: `src/quantlab/ui/market_data.py` 的 `list_market_datasets`
- Create: `tests/conftest.py`、`tests/test_market_catalog.py`
- Extend: `tests/test_ui_market_data.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class MarketVersionRecord:
    symbol: str
    version_id: str
    row_count: int
    imported_at: str
    processed_path: Path
    raw_version_dir: Path
    provider: str
    adjustment: str | None
    metadata_sha256: str

@dataclass(frozen=True)
class CatalogScan:
    records: tuple[MarketVersionRecord, ...]
    issues: tuple[str, ...]

def scan_market_catalog(processed_dir: Path, raw_archive_dir: Path) -> CatalogScan: ...
def read_market_version(record: MarketVersionRecord) -> tuple[pd.DataFrame, str]: ...
```

这里的第二个返回值是读取到的 Parquet 原始字节 SHA-256。`provider` 对
联网数据使用已有 key；文件导入使用 `source_format.json` 的来源 key，
没有该文件时使用 `csv`。CSV 的 `adjustment=None`，不能默认成不复权。

- [x] **Step 1：建立合成数据与真实导入夹具，并写目录测试。**

`tests/conftest.py` 使用如下完整夹具，不写入项目私人数据目录：

```python
import pandas as pd
import pytest
from quantlab.data.importer import import_market_csv

@pytest.fixture
def sample_assets():
    def frame(symbol, dates):
        return pd.DataFrame({
            "date": pd.to_datetime(dates), "symbol": symbol,
            "open": 3.0, "high": 3.2, "low": 2.9,
            "close": 3.1, "volume": 1000.0,
        })
    return {
        "510300": frame("510300", ["2025-01-02", "2025-01-03", "2025-01-07"]),
        "159915": frame("159915", ["2025-01-03", "2025-01-06", "2025-01-07"]),
    }

@pytest.fixture
def imported_pair(tmp_path, sample_assets):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    results = []
    for symbol, data in sample_assets.items():
        source = tmp_path / f"{symbol}.csv"
        data.to_csv(source, index=False)
        results.append(import_market_csv(source, raw, processed))
    return raw, processed, tuple(results)
```

`tests/test_market_catalog.py` 的首个测试：

```python
from quantlab.data.catalog import scan_market_catalog, read_market_version

def test_catalog_keeps_each_selected_version(imported_pair):
    raw, processed, imports = imported_pair
    scan = scan_market_catalog(processed, raw)
    assert not scan.issues
    assert {r.version_id for r in scan.records} == {r.version.version_id for r in imports}
    assert {r.symbol for r in scan.records} == {"510300", "159915"}
    for record in scan.records:
        data, digest = read_market_version(record)
        assert record.adjustment is None
        assert data.symbol.unique().tolist() == [record.symbol]
        assert len(digest) == 64
```

再加入下列参数化测试，分别修改临时版本中的 JSON：无提交标记、坏 JSON、
同一版本多份提交标记、`symbol` 错配、`processed_path` 指向目录外、符号链接
逃出根目录、Provider 版本号/路径错配、同一 ETF 两个版本均保留。每例断言
坏版本不可选且 `issues` 可定位；健康版本仍可选。元数据在扫描后改变时，
`read_market_version` 必须抛出 `ValueError`，不得套用旧价格口径。

- [x] **Step 2：运行红灯测试。**

```bash
.venv/bin/python -m pytest tests/test_market_catalog.py -q
```

预期新模块尚未实现，导入或具体行为断言失败；确认不是夹具错误。

- [x] **Step 3：实现目录和单次快照读取，再接回旧页面。**

使用现有 `import_result.json` 为完成标记，建立版本到元数据路径的索引；
核对代码目录名、文件 stem、版本号、路径、行数及有时区的导入时间。
已知 Provider 必须为日线且调整类型在 `none/qfq/hfq` 内。缺少必需元数据
或身份冲突的版本不发布。路径先 `resolve()` 并确认位于传入根目录内。
`metadata_sha256` 对构成目录身份的元数据字节及固定文件名排序取指纹，
读取选中版本时重新计算，变化即拒绝；重新解析后的路径必须仍等于
扫描时保存的规范绝对路径，防止扫描后被符号链接替换。Parquet 只读取一次：

```python
payload = record.processed_path.read_bytes()
data = pd.read_parquet(BytesIO(payload))
file_sha256 = sha256(payload).hexdigest()
```

读取后核对文件内只有对应字符串代码、行数正确，返回深拷贝；完整行情
数值校验由 Task 3 的规范化执行。扫描排序按导入时间倒序，同时间再按代码、
版本排序。旧包装仍返回原有 `MarketDatasetRecord`，字段与类型不变；
新页面直接用 `CatalogScan.issues`，旧包装将问题记到现有日志且继续返回健康记录。

- [x] **Step 4：验证新增目录与旧数据功能。**

```bash
.venv/bin/python -m pytest tests/test_market_catalog.py tests/test_ui_market_data.py tests/test_market_service.py tests/test_ifind_market_service.py -q
```

预期全部离线测试通过。不得为兼容新目录而取消既有版本完整性测试。

- [x] **Step 5：只审阅并提交该任务的差异。**

```bash
git diff -- src/quantlab/data/catalog.py src/quantlab/ui/market_data.py tests/conftest.py tests/test_market_catalog.py tests/test_ui_market_data.py
```

旧文件含已有 iFinD 改动，须按变更块暂存，仅加入当前任务内容；新文件可
按上述精确路径暂存。确认暂存差异后提交 `feat: add version-aware market catalog`。

## Task 2：建立可解释的日历输入与日期轴

**作用：** 区分“行情里出现过的日期”与“用户提供的完整覆盖声明”。

**Files:**

- Create: `src/quantlab/data/calendar.py`、`tests/test_calendar.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CalendarInput:
    dates: tuple[date, ...]
    source: str
    coverage_start: date
    coverage_end: date
    file_sha256: str | None
    dates_sha256: str

@dataclass(frozen=True)
class CalendarAxis:
    dates: tuple[date, ...]  # 本次实际使用日期
    mode: str              # observed_union / supplied
    source: str
    coverage_start: date
    coverage_end: date
    file_sha256: str | None
    input_dates_sha256: str

def normalize_daily_dates(values: pd.Series) -> pd.Series: ...
def parse_calendar_csv(content: bytes, *, source: str,
                       coverage_start: date, coverage_end: date) -> CalendarInput: ...
def resolve_calendar(observed_dates: tuple[date, ...], *, start: date, end: date,
                     calendar: CalendarInput | None = None) -> CalendarAxis: ...
```

`normalize_daily_dates` 返回无时区、午夜的 pandas 日期列；空值、数值日期、
盘中时间、带时区值、无效日期均拒绝，不使用自动截断。日期重复由调用者
按其键检查。`CalendarInput` 可程序化构造，但 `resolve_calendar` 必须重新
验证其字段、排序唯一性和指纹，不能假设 dataclass 自带验证。

- [x] **Step 1：写并集、显式日历和冲突测试。**

```python
from datetime import date
import pytest
from quantlab.data.calendar import parse_calendar_csv, resolve_calendar

def test_explicit_calendar_preserves_day_missing_from_all_assets():
    calendar = parse_calendar_csv(
        b"date\n2025-01-02\n2025-01-03\n2025-01-06\n",
        source="synthetic fixture", coverage_start=date(2025, 1, 1),
        coverage_end=date(2025, 1, 7),
    )
    axis = resolve_calendar((date(2025, 1, 2), date(2025, 1, 6)),
                            start=date(2025, 1, 1), end=date(2025, 1, 7),
                            calendar=calendar)
    assert axis.mode == "supplied"
    assert date(2025, 1, 3) in axis.dates

@pytest.mark.parametrize("content", [
    b"date\n2025-01-02\n2025-01-02 00:00:00\n",
    b"date\n2025-01-02 13:00:00\n", b"date\n2025-01-02T00:00:00+08:00\n",
    b"date\n", b"date,other\n2025-01-02,x\n",
])
def test_calendar_rejects_ambiguous_or_incomplete_rows(content):
    with pytest.raises(ValueError):
        parse_calendar_csv(content, source="fixture",
                           coverage_start=date(2025, 1, 1),
                           coverage_end=date(2025, 1, 7))
```

加入 UTF-8 BOM 可接受、空来源/无效覆盖声明被拒绝、研究范围超出声明被
拒绝、假日端点不要求有日期行、行情日期不在日历时冲突报错、并集排序
且不添加工作日、字节顺序不同的日历有相同规范日期指纹等测试。

- [x] **Step 2：验证测试先失败。**

```bash
.venv/bin/python -m pytest tests/test_calendar.py -q
```

- [x] **Step 3：实现纯内存校验。**

CSV 用 `utf-8-sig` 解码且只接受 `date` 列；校验再排序，不删除重复。
指纹采用排序后 ISO 日期列表的规范 JSON。`resolve_calendar` 在研究区间
内进行集合差检查；发现原行情日期不在日历时，错误列出具体日期。
默认模式只使用区间内观测日期并集，不能使用 `bdate_range`。

```python
conflicts = set(observed_in_range) - set(calendar_dates_in_range)
if conflicts:
    raise ValueError(f"行情日期不在所提供日历内：{sorted(conflicts)}")
```

显式日历输入为空或区间内无日历日期时停止。无论日期轴模式如何，Task 3
负责拒绝所有资产在研究区间内都无行情的请求。

- [x] **Step 4：验证后提交。**

```bash
.venv/bin/python -m pytest tests/test_calendar.py -q
git add -- src/quantlab/data/calendar.py tests/test_calendar.py
git commit -m "feat: add explicit research calendars"
```

## Task 3：实现纯对齐与数据质量报告

**作用：** 生成统一长表，保证缺失可见且不改变真实行情。

**Files:**

- Create: `src/quantlab/data/alignment.py`、`tests/test_alignment.py`
- Consumes: Task 2 的 `normalize_daily_dates`、`resolve_calendar`、日历类型

**Interfaces:**

```python
@dataclass(frozen=True)
class AlignmentResult:
    data: pd.DataFrame
    coverage: pd.DataFrame
    calendar: CalendarAxis
    common_dates: tuple[date, ...]
    warnings: tuple[str, ...]
    content_sha256: str

def normalize_market_frame(data: pd.DataFrame, symbol: str) -> pd.DataFrame: ...
def align_market_data(datasets: Mapping[str, pd.DataFrame], *, start: date, end: date,
                      calendar: CalendarInput | None = None) -> AlignmentResult: ...
def summarize_alignment(data: pd.DataFrame) -> tuple[pd.DataFrame, tuple[date, ...]]: ...
def validate_aligned_data(data: pd.DataFrame, axis: CalendarAxis,
                          symbols: tuple[str, ...]) -> None: ...
def alignment_content_sha256(data: pd.DataFrame, axis: CalendarAxis) -> str: ...
```

`coverage` 列固定为 `symbol, first_observed, last_observed, calendar_count,
observation_count, missing_count, missing_ratio, longest_missing_run,
zero_volume_count`，无观测时首末日期为空。版本和来源在 Task 4 加入展示
报告，不令纯计算模块依赖目录类型。

- [x] **Step 1：写可手算的对齐测试。**

```python
from datetime import date
import pandas as pd
from quantlab.data.alignment import align_market_data

def test_alignment_keeps_gaps_and_original_values(sample_assets):
    before = {s: d.copy(deep=True) for s, d in sample_assets.items()}
    result = align_market_data(sample_assets, start=date(2025, 1, 2), end=date(2025, 1, 7))
    assert len(result.data) == 8
    absent = result.data.query("symbol == '510300'").set_index("date").loc["2025-01-06"]
    assert not bool(absent.has_observation)
    assert absent[["open", "high", "low", "close", "volume"]].isna().all()
    assert result.common_dates == (date(2025, 1, 3), date(2025, 1, 7))
    row = result.coverage.set_index("symbol").loc["510300"]
    assert row.missing_count == 1
    assert row.missing_ratio == 0.25
    for symbol in sample_assets:
        pd.testing.assert_frame_equal(before[symbol], sample_assets[symbol])
```

追加参数化输入：少于两只、数字代码、空表、空字段、非数字、正负无穷、
重复日期、非法 OHLC、负成交量、不同代码混入，分别断言拒绝；字符串
代码 `000001` 在纯计算测试中保留前导零（只是代码格式测试，不声称其为 ETF）。
另测：乱序与重排行顺序不改变内容指纹、零成交量仍是观测、全区间缺失
的一只仍保留、全部无行情拒绝、没有共同日期警告、连续缺失按日期轴
计数、显式日历增加全体缺失日。

- [x] **Step 2：运行红灯测试。**

```bash
.venv/bin/python -m pytest tests/test_alignment.py -q
```

- [x] **Step 3：实现规范化、重建索引、统计与验证。**

先对每个非空完整输入深拷贝、转换日期和五个数值列、校验有限数值与
旧 `validate_market_data` 的价格逻辑，再按 `(date,symbol)` 查重。之后
裁剪区间；原版本有效但裁剪为空的资产允许保留。核心对齐使用：

```python
grid = pd.MultiIndex.from_product(
    [pd.to_datetime(axis.dates), sorted(datasets)], names=["date", "symbol"],
)
observed = pd.concat(clipped_frames, ignore_index=True).set_index(["date", "symbol"])
observed["has_observation"] = True
aligned = observed.reindex(grid)
aligned["has_observation"] = aligned["has_observation"].eq(True)
aligned = aligned.reset_index()
```

只固定输出所需八列，缺失数值保留 NaN。`validate_aligned_data` 要验证
完整笛卡尔积、日期/代码顺序与唯一性、布尔观测列、缺失行全空且真实
观测行全部有效；不可把缺失行交给旧的无空值校验。
`summarize_alignment` 按布尔连续段计算最长缺失，按日期的全部观测条件
得到共同日期。规范 JSON 将日期转 ISO、缺失转 `null`、浮点保留精度，
`allow_nan=False`，连同日历轴声明计算内容指纹；不导入回测模块的工具。

- [x] **Step 4：绿灯验证并提交。**

```bash
.venv/bin/python -m pytest tests/test_calendar.py tests/test_alignment.py tests/test_validator.py -q
git add -- src/quantlab/data/alignment.py tests/test_alignment.py
git commit -m "feat: align ETF observations without filling gaps"
```

## Task 4：固定选择与价格口径，生成研究集合预览

**作用：** 把“选中了哪些版本”与“实际计算用了什么”绑定在一起。

**Files:**

- Create: `src/quantlab/data/research_sets.py`、`tests/test_research_sets.py`
- Extend: `tests/conftest.py`（新增研究预览夹具）

**Interfaces:**

```python
@dataclass(frozen=True)
class DatasetSelection:
    record: MarketVersionRecord
    declared_adjustment: str | None = None

@dataclass(frozen=True)
class ResearchSetPreview:
    alignment: AlignmentResult
    sources: tuple[dict, ...]  # 下文白名单字段，按 symbol 排序
    start: date
    end: date
    adjustment: str
    coverage: pd.DataFrame  # 核心 coverage 一对一加入版本/来源/口径依据
    warnings: tuple[str, ...]
    preview_sha256: str

def build_research_set(selections: tuple[DatasetSelection, ...], *, start: date,
                       end: date, calendar: CalendarInput | None = None) -> ResearchSetPreview: ...
def validate_research_preview(preview: ResearchSetPreview) -> None: ...
def research_preview_sha256(preview: ResearchSetPreview) -> str: ...
```

`sources` 仅含 `symbol, version_id, provider, imported_at, processed_path,
file_sha256, metadata_sha256, input_content_sha256, adjustment,
adjustment_basis`；依据为 `provider` 或 `user_declared`。内容指纹对规范化
后的完整单 ETF 输入计算，保存的对齐快照仅含区间内所需行情。
`preview_sha256` 对 alignment 内容指纹、区间、共同口径及 sources 的
非路径字段取规范指纹，不包含自身、当前时间或保存 ID。

- [x] **Step 1：写口径和版本绑定测试。**

```python
from datetime import date
import pytest
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.research_sets import DatasetSelection, build_research_set

def test_csv_adjustment_must_be_declared(imported_pair):
    raw, processed, _ = imported_pair
    records = scan_market_catalog(processed, raw).records
    with pytest.raises(ValueError, match="复权|口径"):
        build_research_set(tuple(DatasetSelection(r) for r in records),
                           start=date(2025, 1, 2), end=date(2025, 1, 7))
    result = build_research_set(tuple(DatasetSelection(r, "none") for r in records),
                                start=date(2025, 1, 2), end=date(2025, 1, 7))
    assert result.adjustment == "none"
    assert {s["version_id"] for s in result.sources} == {r.version_id for r in records}
    assert {s["adjustment_basis"] for s in result.sources} == {"user_declared"}
```

增加混合口径、重复代码不同版本、试图覆盖 Provider 的已知口径、来源
不同保留并警告的测试。创建新版本后旧选择仍指向旧版本；改变输入顺序
不改变预览指纹。修改预览的 data、sources 或 coverage 后，调用
`validate_research_preview` 必须报错。

新增后续任务可用的夹具（在夹具内部导入新模块）：

```python
@pytest.fixture
def research_preview(imported_pair):
    from datetime import date
    from quantlab.data.catalog import scan_market_catalog
    from quantlab.data.research_sets import DatasetSelection, build_research_set
    raw, processed, _ = imported_pair
    selected = tuple(DatasetSelection(r, "none") for r in scan_market_catalog(processed, raw).records)
    return build_research_set(selected, start=date(2025, 1, 2), end=date(2025, 1, 7))
```

- [x] **Step 2：确认红灯。**

```bash
.venv/bin/python -m pytest tests/test_research_sets.py -q
```

- [x] **Step 3：连接目录快照与纯对齐。**

先验证选择唯一性和声明值，再逐只使用 `read_market_version`。已知
口径不允许不同的人工声明；未知口径必须显式提供。统一口径后调用
Task 3 的规范化及对齐，不重复实现对齐逻辑。报告通过 `symbol` 做
`validate="one_to_one"` 合并，并给未知转声明、混合来源、无共同日期等
可解释警告。

```python
basis = "provider" if record.adjustment is not None else "user_declared"
if record.adjustment is not None and declared not in (None, record.adjustment):
    raise ValueError("不能覆盖数据源记录的复权口径")
adjustment = record.adjustment if record.adjustment is not None else declared
if adjustment not in {"none", "qfq", "hfq"}:
    raise ValueError("请确认该版本的复权口径")
```

`validate_research_preview` 重验长表、日历、覆盖统计和所有声明；将重算
内容与既有指纹比对，不能以重新赋值指纹来“修复”被修改对象。无区间
观测的资产从 sources 提供身份，统计允许它有 100% 缺失。

- [x] **Step 4：验证并提交。**

```bash
.venv/bin/python -m pytest tests/test_research_sets.py tests/test_market_catalog.py tests/test_alignment.py -q
git add -- src/quantlab/data/research_sets.py tests/test_research_sets.py tests/conftest.py
git commit -m "feat: build version-pinned ETF research sets"
```

## Task 5：可靠保存和重读研究集合

**作用：** 让已验证的研究数据不依赖原文件位置，且保存失败不冒充成功。

**Files:**

- Create: `src/quantlab/data/research_storage.py`、`tests/test_research_storage.py`
- Modify: `.gitignore`（仅增加 `data/research_sets/`）

**Interfaces:**

```python
@dataclass(frozen=True)
class SavedResearchSet:
    set_id: str
    name: str
    created_at: str
    directory: Path

@dataclass(frozen=True)
class ResearchSetScan:
    records: tuple[SavedResearchSet, ...]
    issues: tuple[str, ...]

@dataclass(frozen=True)
class LoadedResearchSet:
    record: SavedResearchSet
    preview: ResearchSetPreview

def save_research_set(preview: ResearchSetPreview, output_dir: Path, *, name: str) -> SavedResearchSet: ...
def list_research_sets(output_dir: Path) -> ResearchSetScan: ...
def load_research_set(output_dir: Path, set_id: str) -> LoadedResearchSet: ...
```

- [ ] **Step 1：写保存、重读与损坏测试。**

```python
from pathlib import Path
import pandas as pd
import pytest
from quantlab.data.research_storage import save_research_set, load_research_set, list_research_sets

def test_saved_set_does_not_need_original_files(tmp_path, research_preview):
    root = tmp_path / "sets"
    saved = save_research_set(research_preview, root, name="合成示例")
    for source in research_preview.sources:
        path = Path(source["processed_path"])
        path.rename(path.with_suffix(".moved"))  # 仅移动本测试 tmp_path 中的合成文件
    restored = load_research_set(root, saved.set_id)
    pd.testing.assert_frame_equal(restored.preview.alignment.data, research_preview.alignment.data)
    assert restored.preview.preview_sha256 == research_preview.preview_sha256
    assert len(list_research_sets(root).records) == 1

def test_tampered_file_is_rejected(tmp_path, research_preview):
    saved = save_research_set(research_preview, tmp_path / "sets", name="示例")
    (saved.directory / "coverage.csv").write_text("altered", encoding="utf-8")
    with pytest.raises(ValueError, match="指纹|损坏"):
        load_research_set(tmp_path / "sets", saved.set_id)
```

另测：保存前预览被修改则无新目录、空名称被拒绝、连续保存 ID 不同、
UUID 碰撞不覆盖（替换 storage 模块的 uuid4）、写 Parquet 或 manifest
失败时不出现在列表、坏 manifest 隔离、非法 set_id 和符号链接越界拒绝。
用假凭证环境值验证产物不含该值。修改文件并同步修改其文件哈希时，
仍需通过内容指纹、覆盖重算、输入快照语义校验，不能只验证字节哈希。

- [ ] **Step 2：运行红灯测试。**

```bash
.venv/bin/python -m pytest tests/test_research_storage.py -q
```

- [ ] **Step 3：实现只追加的快照保存。**

保存之前校验预览；名称去首尾空白但不作为目录名。生成 UUID，独占
创建目录，写 `aligned.parquet` 与 `coverage.csv`，覆盖表 CSV 读写明确
代码为字符串、日期为空时的规则和浮点回读精度。manifest 白名单包含
Spec 的身份、输入、日历、版本和限制字段，以及两份输出的文件指纹。

```python
destination = output_dir / str(uuid4())
destination.mkdir(parents=True, exist_ok=False)
# 数据写入并计算指纹后，才发布完成标记。
temporary_manifest = destination / ".manifest.tmp"
temporary_manifest.write_text(serialized_manifest, encoding="utf-8")
temporary_manifest.replace(destination / "manifest.json")
```

JSON 使用 `allow_nan=False`，NaT/NaN 在元数据中转换为 `null`。仅发布
完整 manifest；中断遗留目录保持不可见，不自动删除其他用户目录。
输出必须与 `ResearchSetPreview` 深拷贝出的保存快照一致，避免保存中
调用方改动可变 DataFrame 导致跨文件不一致。

`load_research_set` 只接受规范 UUID，从固定根目录定位固定文件名，
不执行 manifest 内的绝对路径。校验版本、文件指纹后，恢复明确类型，
调用纯长表语义验证和 coverage 重算，再比较预览指纹。不能重新访问
sources 中的原路径，也不因读取而重新选择行情版本。列表对半成品
忽略，对损坏完成标记记录 issue，排序按 UTC 创建时间倒序。

- [ ] **Step 4：测试、检查忽略规则并提交。**

```bash
.venv/bin/python -m pytest tests/test_research_storage.py tests/test_research_sets.py -q
git check-ignore data/research_sets/example/manifest.json
git add -- src/quantlab/data/research_storage.py tests/test_research_storage.py
```

`.gitignore` 已有阶段 1 未提交内容，只暂存新增研究集合这一行。确认
暂存差异后提交 `feat: persist auditable ETF research snapshots`。

## Task 6：接入页面并完成本小步验收

**作用：** 用户无需写代码即可完成选择、对齐检查、保存与重新查看。

**Files:**

- Create: `src/quantlab/ui/multi_asset.py`、`pages/6_多ETF数据.py`
- Modify: `app.py`、`src/quantlab/ui/runtime.py`
- Create: `tests/test_ui_multi_asset.py`、`tests/test_stage2_data_acceptance.py`
- Create: `docs/multi_etf_data.md`
- Update: `README.md`、`docs/roadmap.md`（只更新本小步，不标记整个第二阶段完成）

**Interfaces:**

```python
# ui/runtime.py
def get_research_storage_dir(project_root: str | Path) -> Path: ...

# ui/multi_asset.py
def research_request_sha256(*, name: str, selections: tuple[DatasetSelection, ...],
                            start: date, end: date, calendar: CalendarInput | None) -> str: ...
def can_save_preview(current_request_sha256: str, preview_request_sha256: str | None) -> bool: ...
def friendly_research_error(error: Exception) -> tuple[str, str]: ...
```

数据服务直接从各自模块导入，避免为了新页面扩大旧 `ui/__init__.py` 的
依赖。状态键使用 `research_` 前缀，与第一阶段页面隔离。

- [ ] **Step 1：先写请求变化和页面正常流程测试。**

```python
from datetime import date
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.research_sets import DatasetSelection
from quantlab.ui.multi_asset import research_request_sha256, can_save_preview

def test_changed_request_cannot_save_old_preview(imported_pair):
    raw, processed, _ = imported_pair
    selections = tuple(DatasetSelection(r, "none") for r in scan_market_catalog(processed, raw).records)
    old = research_request_sha256(name="示例", selections=selections,
                                  start=date(2025, 1, 2), end=date(2025, 1, 7), calendar=None)
    changed = research_request_sha256(name="示例", selections=selections,
                                      start=date(2025, 1, 3), end=date(2025, 1, 7), calendar=None)
    assert can_save_preview(old, old)
    assert not can_save_preview(changed, old)
    assert not can_save_preview(old, None)
```

页面测试沿用 `tests/test_ui_run_records.py` 的 AppTest 模式，只替换存储根
目录边界，目录扫描、读取、对齐与保存都跑真实代码。widget key 固定：
`research_name`、`research_symbols`、`research_version_<symbol>`、
`research_adjustment_<symbol>`、`research_start`、`research_end`、
`research_calendar_mode`、`research_preview`、`research_save`、
`research_saved_id`、`research_load`。日期模式值 `observed_union/supplied`
通过中文 `format_func` 展示。页面至少执行此验收片段：

```python
from streamlit.testing.v1 import AppTest

page = AppTest.from_file("pages/6_多ETF数据.py").run()
page.text_input(key="research_name").set_value("合成验收集合")
page.multiselect(key="research_symbols").set_value(["510300", "159915"]).run()
for symbol in ("510300", "159915"):
    page.selectbox(key=f"research_adjustment_{symbol}").set_value("none")
page.button(key="research_preview").click().run()
assert not page.exception
page.button(key="research_save").click().run()
assert not page.exception
assert page.success
```

测试中先将数据根目录替换为 `imported_pair`，研究集合根目录替换为
临时 sets，再断言磁盘只有一个完整集合并点击重读。修改版本、日期、
名称、日历内容任一项后，保存按钮禁用且服务端再次检查请求指纹。
AppTest 暂无上传接口时只替代 `st.file_uploader` 返回合成 BytesIO，
不能 mock 日历解析或对齐结果。

- [ ] **Step 2：确认红灯，再实现页面服务和目录隔离。**

```bash
.venv/bin/python -m pytest tests/test_ui_multi_asset.py -q
```

`get_research_storage_dir` 通过现有 `get_market_storage_dirs` 返回根目录：

```python
raw_dir, _ = get_market_storage_dirs(project_root)
return raw_dir.parent.parent / "research_sets"
```

这样本地落在 data 下，公开模式落在既有会话临时根下；不可另建全局共享
临时目录。测试两份独立 session_state 容器返回不同路径且互相不可见，
并验证同一会话页面重跑仍复用原临时根。

请求指纹包含名称、规范排序后的版本身份及元数据指纹、每只口径声明、
日期、日历内容及覆盖/来源，不能用 Python 的随机化 `hash()`。只在点击
预览时运行生成，正常 widget 放在 form 外以便变更时立即使旧预览失效。
保存时从 session_state 取得匹配请求的完整预览，拒绝缺少或不匹配状态；
不在保存时重新选择“最新”版本。若源文件在预览后改变，保存已校验的
预览快照，明确保存的是本次预览；新预览则重新读取数据并重新记录指纹。

错误分 `ValueError` 数据/声明、`FileNotFoundError` 版本缺失、`OSError`
保存/读取失败，页面给出自然语言，不显示堆栈。无数据不崩溃而引导
已有导入页面；没有共同日期等警告不伪装为失败。写入失败不显示成功，
预览继续保留。

- [ ] **Step 3：完成导航、显式日历、保存列表和失败路径。**

在 `app.py` 的“我的投资”中添加 `st.Page`，名称“多 ETF 数据”。
页面顶部说明研究集合不等于可交易股票池。日历模式用单选，上传模式
要求来源与覆盖范围；来源和 coverage 放在预览中可检查。
已保存列表使用 `list_research_sets`，点击“载入”运行完整校验；发现损坏
时呈现提示，不能继续展示前一个集合并将其冒充为当前选择。

测试覆盖两个日历模式、保存失败、损坏集合、目录空、共同日期空、
同次预览重复保存生成不同 ID、来源声明警告以及隐私隔离。

- [ ] **Step 4：完成真实链路测试与使用说明。**

`tests/test_stage2_data_acceptance.py` 不通过页面 mock 核心流程：用
`imported_pair` → 目录 → 明确口径 → 显式日历增加一个全体缺失日 →
生成 → 保存 → 移动原行情目录 → 重读 → 验证 2 × 日期数行、缺失与
共同日期不变。测试中 ETF 数据和日历均注明为合成示例。

`docs/multi_etf_data.md` 写出六步页面操作、各模块为何存在、并集不能
发现全体缺失日、声明不等于认证、零成交量不等于无记录、缺失不等于
停牌、历史股票池限制、本地与临时保存差异。README 添加入口；路线图
仅将“多 ETF 数据管理和交易日对齐”标为完成，因子与组合仍未实施。

- [ ] **Step 5：运行完整验证并手动检查页面。**

```bash
.venv/bin/python -m pytest -q -rs
.venv/bin/python -m pip check
git diff --check
```

测试退出码必须为 0，跳过项明确列出，不能沿用历史测试数量。使用隔离
的示例数据目录进行浏览器检查，避免导入假数据污染私人库；实际查看
并集警告、上传日历导致全体缺失、保存反馈与重读表格。可暂用既有公开
会话模式并仅绑定本机进行隔离，不创建外网分享链接。无浏览器验证条件
时明确记录限制，不能声称完成视觉验收。

- [ ] **Step 6：最终差异审查、精确提交和交接。**

逐一检查本任务新文件及 app/runtime 改动。README、roadmap 已有收尾
改动，按变更块暂存当前任务内容，保留其他改动。仅提交已验证的本任务
文件，提交信息 `feat: expose multi-ETF research data workflow`。
随后按选定执行方式完成代码审查，发现问题时先补测试再修复并重新验证。
向用户报告本小步已实现内容、实际测试结果、限制及下一步，不将整个
第二阶段标为完成。

## 设计覆盖自查

| 设计要求 | 对应任务 |
|---|---|
| 既有流程复用、固定选择、源文件不覆盖 | 1、4 |
| 来源与复权口径明确，未知必须声明 | 1、4、6 |
| 日期并集与导入日历、覆盖声明及冲突 | 2、3、6 |
| 无填充、缺失和零成交量分开、无状态臆测 | 3、6 |
| 首末观测、连续缺失、共同日期质量报告 | 3、4 |
| 版本及输入/输出指纹、半成品隔离、独立重读 | 4、5 |
| 本地/临时会话隔离、不读取凭证和交易数据 | 5、6 |
| 原有接口兼容与第一阶段回归 | 1、6 |
| 用户逐步学习与模块解释 | 各 Task 的作用说明、6 的使用文档 |

## 执行方式建议

推荐由当前助手在本任务中逐步执行（Native），每完成一个任务解释结果，
最后做独立代码审查。六个任务高度依赖前面的类型与接口，连续实施便于
保持一致，也符合用户逐步学习的要求。也可采用每步独立子代理实现和
审查的方式，但需要更多上下文交接。

本计划需用户审阅并选择执行方式后开始。确认计划不代表已有任何新增
业务代码或测试通过；执行记录以实际完成的复选框和工具证据为准。
