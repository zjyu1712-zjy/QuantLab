# QuantLab

QuantLab 是一个本地运行、以 Python 为主、面向个人投资者的量化投资
成长平台。它帮助用户通过数据分析、策略实验和交易复盘，逐步建立科学
投资体系。

项目不连接券商，不包含自动下单、实盘交易、实时行情或盘中交易系统。

## 当前范围

本项目当前实施：

- 阶段 0：架构与规则统一；
- 第一阶段：单只 ETF 的分析、回测与交易复盘 MVP。

多 ETF、因子研究和多因子组合只记录在路线图中，当前不实施。

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
cd "/Users/zhoujiayu/Documents/量化/QuantLab"
source .venv/bin/activate
```

运行全部测试：

```bash
PYTHONPATH=src python -m pytest -v
```

启动本地可视化界面：

```bash
python -m streamlit run app.py
```

启动后浏览器会打开 QuantLab。页面按照“我的投资 → 分析 → 实验 → 改进”
组织，用户可以添加 ETF、了解收益和风险、验证投资方法，再把真实交易
与规则信号进行复盘比较。

在“添加资产数据”中输入 ETF 代码和日期即可自动获取沪深 ETF 日线行情。
系统会先检查本地缓存，只补充缺少的首尾区间，并把不同复权方式分别保存。
CSV 上传仍作为行情服务不可用时的备用入口。

## iFinD QuantAPI

QuantLab 可通过服务端配置选择行情来源：

- `MARKET_DATA_PROVIDER=auto`：配置了 iFinD 凭证时使用 iFinD，否则使用 AkShare；
- `MARKET_DATA_PROVIDER=akshare`：始终使用 AkShare；
- `MARKET_DATA_PROVIDER=ifind`：始终使用 iFinD，未配置凭证时会给出简短提示。

`auto` 只在请求前根据配置选择数据源。一旦选择 iFinD，请求失败时会
如实提示，不会在用户不知情的情况下切换到 AkShare。

iFinD refresh token 只应保存在本机或网站服务器的环境变量中，不要填入
浏览器页面、写入源代码或提交到代码仓库。项目根目录中的 `.env.example`
只列出配置名称，不包含真实凭证。

正常本地启动：

```bash
cd /Users/zhoujiayu/Documents/量化/QuantLab
source .venv/bin/activate
PYTHONPATH=src streamlit run app.py
```

只有在已获得 iFinD 授权、并且已由终端或服务器环境安全提供
`IFIND_REFRESH_TOKEN` 时，才手动运行真实接口检查：

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
├── config/       默认配置和日志初始化
├── data/         行情服务、数据源、缓存和交易数据处理
└── factors/      第二阶段因子扩展占位

tests/            自动测试
docs/             数据、指标、回测假设和路线图
logs/             本地日志（日志内容不纳入版本管理）
data/raw/         原始私人数据
data/processed/   标准化私人数据
```

## 规则文档

- [数据格式](docs/data_formats.md)
- [指标定义](docs/metrics.md)
- [回测假设](docs/backtest_assumptions.md)
- [开发路线图](docs/roadmap.md)
