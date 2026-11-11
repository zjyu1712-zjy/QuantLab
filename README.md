# QuantLab｜量化研究与交易复盘

面向金融学习与策略研究的本地 Python / Streamlit 平台：从 ETF 数据整理、指标分析，到策略回测、组合实验和交易复盘。

**仅用于研究与教育。没有券商连接或自动下单功能，不构成投资建议。回测成绩不代表未来表现。**

## 可以做什么

| 模块 | 当前能力 |
| --- | --- |
| 数据管理 | CSV 导入、数据校验、版本化存储、AkShare / 可选 iFinD 接口 |
| 单资产研究 | 收益、波动率、回撤、夏普比率、买入持有与双均线对比 |
| 交易复盘 | FIFO 匹配、交易盈亏、持仓资金曲线与规则信号比较 |
| 多 ETF 研究 | 日期对齐、固定版本研究集合、因子排名与评分 |
| 组合实验 | Top N 等权、成本与调仓、基准对比、档案和 ZIP 导出 |
| 诊断 | 分年份表现、收益集中度、样本内参数对比与行情分组 |

## 快速开始

需要 Python 3.12。仓库保留 `QuantLab/` 应用子目录。

```bash
git clone https://github.com/zjyu1712-zjy/QuantLab.git
cd QuantLab/QuantLab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-tested.txt
python -m streamlit run app.py
```

Windows 使用 `.venv\Scripts\activate` 激活环境。应用使用本地磁盘保存数据，请勿直接部署到公开网络处理私人交易记录。

### 不需要行情账号的演示

```bash
# 在仓库根目录执行；生成明确标注的虚构数据，不访问网络。
python3 scripts/generate_demo_data.py
```

在应用“添加资产数据”中上传 `examples/synthetic_etf.csv`，交易复盘中使用 `examples/synthetic_trades.csv`。这些价格和交易均为人工生成，不代表任何真实证券或个人账户。演示标的代码仅为格式占位。

### 验证

```bash
cd QuantLab
PYTHONPATH=src RUN_IFIND_INTEGRATION=0 python -m pytest -q
```

普通测试使用本地夹具或模拟接口。商业行情真实接口测试默认跳过；请勿在 CI 中填写私人 token。验证范围与结果见[发布检查](docs/PUBLICATION.md)。

## 文档入口

- [应用与行情配置](QuantLab/README.md)
- [数据格式](QuantLab/docs/data_formats.md)
- [指标定义](QuantLab/docs/metrics.md)
- [回测假设](QuantLab/docs/backtest_assumptions.md)
- [因子研究](QuantLab/docs/factors.md)
- [组合回测](QuantLab/docs/portfolio_backtest.md)
- [回测记录](QuantLab/docs/backtest_records.md)
- [路线图与未实现范围](QuantLab/docs/roadmap.md)
- [数据、隐私与使用边界](DATA_AND_PRIVACY.md)

## 研究边界

信号与成交时点、手续费、滑点、复权口径、研究池选择都会影响结果。当前参数对比属于样本内分析，不能据此断言策略可泛化。固定动量与波动率双因子评分只作观察；并非已实现完整多因子投资系统。

本仓库不包含私人交易流水、行情缓存、密钥、商业数据下载包或原虚拟环境。外部行情数据由使用者自行取得合法授权。

## 权利说明

仓库公开供展示和审阅，目前未新增开源许可证，也不授予超出适用法律及 GitHub 平台条款的使用权。依赖库遵循各自许可证，数据源遵循各自服务条款。商业使用、再分发代码或数据前，请先确认相应权利。
