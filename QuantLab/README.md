# QuantLab

QuantLab 是一个本地运行、以 Python 为主、面向个人投资者的量化投资
成长平台。它帮助用户通过数据分析、策略实验和交易复盘，逐步建立科学
投资体系。

项目不连接券商，不包含自动下单、实盘交易、实时行情或盘中交易系统。

## 当前范围

当前代码包括：

- 单 ETF 数据导入、收益风险分析、双均线回测与 FIFO 交易复盘；
- 多 ETF 数据版本管理、交易日期对齐和研究集合保存；
- 动量、波动率、均线偏离、最大回撤、成交量变化和趋势效率因子；
- 单因子排名、相对评分、Top N 等权组合回测及全池买入持有基准；
- 回测档案、年份表现、收益集中度、样本内参数对比和研究池行情分组。

双因子综合评分目前用于观察，不参与组合调仓。样本外检验、更完整的参数
稳定性研究和真实行情授权验证仍需继续完善。详见[开发路线图](docs/roadmap.md)。

## 本地运行环境

- Python 3.12
- pandas
- NumPy
- PyArrow
- AkShare（ETF 历史行情）
- pytest
- Streamlit

进入项目并激活环境：

```bash
cd QuantLab/QuantLab
source .venv/bin/activate
```

运行全部测试：

```bash
PYTHONPATH=src python -m pytest -v
```

`pytest.ini` 会自动加入 `src` 导入路径，其中子进程测试仍需设置 `PYTHONPATH=src`。

启动本地可视化界面：

```bash
python -m streamlit run app.py
```

启动后浏览器会打开 QuantLab。页面按照“我的投资 → 分析 → 实验 → 改进”
组织，用户可以添加 ETF、了解收益和风险、验证投资方法，再把真实交易
与规则信号进行复盘比较。

在“添加资产数据”中输入 ETF 代码和日期即可自动获取沪深 ETF 日线行情。
系统会先检查本地缓存，只补充所有尚未覆盖的日期区间，并把不同复权方式分别保存。
CSV 上传仍作为行情服务不可用时的备用入口。

## iFinD QuantAPI

QuantLab 可通过服务端配置选择行情来源：

- `MARKET_DATA_PROVIDER=auto`：配置了 iFinD 凭证时使用 iFinD，否则使用 AkShare；
- `MARKET_DATA_PROVIDER=akshare`：始终使用 AkShare；
- `MARKET_DATA_PROVIDER=ifind`：始终使用 iFinD，未配置凭证时会给出简短提示。

`auto` 只在请求前根据配置选择数据源。一旦选择 iFinD，请求失败时会
如实提示，不会在用户不知情的情况下切换到 AkShare。

iFinD access token 只应保存在本机或网站服务器的环境变量中，不要填入
浏览器页面、写入源代码或提交到代码仓库。项目根目录中的 `.env.example`
只列出配置名称，不包含真实凭证。

QuantLab 新启动流程直接使用 `IFIND_ACCESS_TOKEN` 请求行情，不会调用
`/api/v1/get_access_token`，也不会在 access token 被拒绝时自动回退到
refresh token。旧的 refresh-token 交换能力仅为兼容已有部署而保留。

### macOS 安全保存与启动（推荐）

首次使用时，在 QuantLab 项目目录运行：

```bash
./configure_ifind_mac.command
```

macOS 钥匙串会提示输入密码。把 iFinD `access token` 粘贴进去并确认；
输入内容不会显示，QuantLab 也不会把凭证写入项目文件。以后即使关闭终端
或重新启动电脑，也不需要再次手动执行 `export`。

以后启动 QuantLab 只需运行：

```bash
./run_ifind_mac.command
```

该命令会在运行时从 macOS 钥匙串读取凭证，并只把凭证提供给本次启动的
QuantLab 进程。如果 access token 过期或需要更换，重新运行
`./configure_ifind_mac.command` 即可更新同一条钥匙串记录。

这两个命令只适用于 macOS。本机钥匙串方案不用于网站服务器；网站部署应
通过服务器或云平台的 Secret Manager 注入凭证，并且必须先确认 iFinD 授权
允许多用户间接访问、数据展示与缓存。

正常本地启动：

```bash
cd QuantLab/QuantLab
source .venv/bin/activate
PYTHONPATH=src streamlit run app.py
```

只有在已获得 iFinD 授权、并且已由终端或服务器环境安全提供
`IFIND_ACCESS_TOKEN` 时，才手动运行真实接口检查：

```bash
RUN_IFIND_INTEGRATION=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/test_ifind_integration.py -v
```

当前 iFinD 接入的限制：

- 仅支持沪深 ETF 日线历史行情；
- 真实返回形式需使用具有有效权限的 iFinD 账号核验；
- 选定 iFinD 后不会在请求失败时静默切换数据源；
- 如果将 QuantLab 提供给多个网站用户，必须先确认 iFinD 授权允许这种
  服务端使用、数据展示和缓存方式。

未来增加其他商业数据源时，在 `src/quantlab/data/providers/` 下新增实现
`MarketDataProvider` 的 Provider 即可。页面、行情服务、收益分析和回测模块不需
因此改变。

退出虚拟环境：

```bash
deactivate
```

## 项目结构

```text
src/quantlab/
├── analytics/    收益与风险指标
├── backtest/     回测、基准、运行记录与 JSON 保存
├── config/       默认配置和日志初始化
├── data/         行情服务、数据源、缓存和交易数据处理
├── factors/      因子定义、验证、评分与研究
├── portfolio/    多 ETF 组合回测、档案与诊断
├── strategies/   双均线信号
├── trades/       FIFO 匹配和个人交易分析
└── ui/           Streamlit 页面使用的服务

tests/            自动测试
docs/             数据、指标、回测假设和路线图
logs/             本地日志（日志内容不纳入版本管理）
data/raw/         原始私人数据
data/processed/   标准化私人数据
data/backtests/   每次回测的输入、配置和结果记录
```

## 回测运行记录

“策略实验”每次点击运行都会生成独立记录，自动保存在
`data/backtests/<run_id>.json`。“方法对比”会分别保存买入持有和双均线
两个模拟回测，个人交易上传仍只用于当前分析。

在结果中的“实验记录与下载”展开项可以下载同一份 JSON。记录包含数据
来源和版本、实际输入快照及指纹、策略参数、手续费和滑点、指标配置、
UTC 运行时间、代码指纹、Python/数值库版本，以及订单、成交、资金曲线
和绩效摘要。重复点击运行会生成新记录，不覆盖历史结果。

使用 Python API 时，回测返回值自带记录；调用 `save_backtest_run` 才会
写入磁盘。详细用法见[回测记录说明](docs/backtest_records.md)。

阶段 0/1 的收尾检查与阶段 2 交接边界见[验收记录](docs/stage1_acceptance.md)。

## 规则文档

- [数据格式](docs/data_formats.md)
- [指标定义](docs/metrics.md)
- [回测假设](docs/backtest_assumptions.md)
- [开发路线图](docs/roadmap.md)
- [回测运行记录](docs/backtest_records.md)
- [阶段 0/1 收尾验收](docs/stage1_acceptance.md)
