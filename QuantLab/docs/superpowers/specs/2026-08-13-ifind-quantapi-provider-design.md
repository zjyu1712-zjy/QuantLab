# iFinD QuantAPI Provider 接入设计

## 1. 目标与边界

QuantLab 增加一个服务器端 iFinD QuantAPI 行情连接器，使现有
`MarketDataService` 能够通过 iFinD 获取沪深 ETF 历史日线行情。当前
尚未提供正式权限，因此本次实现必须在无真实凭证时可完整测试，且
默认不启用 iFinD。

本次不实现：

- 供外部程序调用 QuantLab 的公共 REST API；
- 用户在页面输入 iFinD 凭证；
- 用户注册、租户管理或计费；
- 绕过 iFinD 权限或数据许可；
- 修改收益分析、风险指标、策略回测和交易复盘逻辑。

## 2. 架构

```text
Streamlit 页面
    ↓
MarketDataService
    ↓
Provider 工厂
    ├── IFindQuantAPIProvider
    │       ↓
    │   IFindQuantAPIClient
    │       ↓
    │   iFinD HTTP QuantAPI
    └── AkShareETFProvider
```

`MarketDataService` 继续作为页面唯一调用入口，负责代码规范化、
日期检查、缓存、数据质量检查和版本保存。新 Provider 只负责调用
iFinD 并返回 QuantLab 的六列标准数据。

## 3. 组件

### 3.1 `IFindQuantAPIClient`

封装 HTTP 通信，不了解 pandas 或 QuantLab 的数据校验。职责包括：

- 使用 `refresh_token` 获取 `access_token`；
- 在内存中缓存尚未过期的 `access_token`；
- 调用 iFinD 历史行情 HTTP 端点；
- 设置请求超时，对短暂性网络错误进行有限次数重试；
- 将 HTTP、认证、限频和响应格式错误转换为 Provider 异常。

客户端支持注入 HTTP transport，以便测试使用模拟响应，不访问真实
iFinD。

### 3.2 `IFindQuantAPIProvider`

实现现有 `MarketDataProvider.fetch_etf_history()` 接口，职责包括：

- 将 `daily` 和 `none/qfq/hfq` 参数转换为 iFinD 查询参数；
- 将 iFinD 响应字段识别并转换为
  `date/open/high/low/close/volume`；
- 在空数据、字段变化和非法响应时抛出稳定的 Provider 异常；
- 不执行项目级缓存和数据版本保存。

### 3.3 Provider 工厂

`get_market_data_provider()` 根据服务器环境变量选择 Provider：

- `MARKET_DATA_PROVIDER=akshare`：只使用 AkShare；
- `MARKET_DATA_PROVIDER=ifind`：只使用 iFinD，缺少凭证时立即给出友好错误；
- `MARKET_DATA_PROVIDER=auto`：当 iFinD 凭证完整时选择 iFinD，否则选择
  AkShare。

第一个实施版本中，`auto` 只在建立 Provider 时选择一次，不在 iFinD
请求失败后静默切换数据源。这能避免在用户不知情时混用不同来源或
复权口径。界面应提示 iFinD 暂时不可用，再由用户选择是否使用备用
数据源。

## 4. 凭证和安全

服务器从环境变量或部署平台的密钥管理功能读取：

```text
MARKET_DATA_PROVIDER=auto
IFIND_REFRESH_TOKEN=[由部署平台密钥管理器提供]
```

实现必须保证：

- 不将 token 写入源码、页面、缓存元数据、错误信息或日志；
- 不将真实凭证提交到 Git；
- 缺少凭证时仍可以启动 QuantLab 和使用 AkShare；
- 页面不提供凭证输入框；
- 在未确认 iFinD 多用户数据授权前，不向公开用户开放该数据源。

## 5. 数据流

1. 页面将 ETF 代码、日期范围和复权方式交给 `MarketDataService`。
2. 服务检查已有缓存并计算缺少区间。
3. 工厂创建的 Provider 只获取缺少区间。
4. iFinD Provider 将响应转换为六列标准数据。
5. `MarketDataService` 复用已有的排序、去重、价格校验和成交量校验。
6. 标准数据按 Provider、复权方式和数据版本分开保存。
7. 页面继续使用现有预览和“开始分析”流程。

## 6. 错误处理

用户页面只显示自然语言：

- 未配置权限：“iFinD 行情服务尚未配置，请联系管理员或使用备用
  行情来源。”
- 认证失败：“iFinD 行情服务认证失败，请联系管理员。”
- 超时或暂时不可用：“行情服务暂时无法连接，请稍后重试。”
- 空数据：“未找到该 ETF 的行情数据，请检查代码或日期范围。”
- 字段变化：“行情服务返回的数据格式暂时无法识别，已记录问题。”

技术细节记入服务器日志，但日志必须先脱敏。不向普通用户显示 Python
traceback。

## 7. 测试

默认测试不访问外部网络，至少覆盖：

- 无凭证时 Provider 不可用且 `auto` 选择 AkShare；
- 明确选择 iFinD 但缺少凭证时返回友好错误；
- access token 获取和有效期内复用；
- 鉴权失效后有限刷新，不无限重试；
- 历史行情请求参数及复权映射；
- iFinD 响应字段转换；
- 空数据、缺失字段、超时、HTTP 错误和限频错误；
- token 不出现在异常文本或日志中；
- Provider 可替换性：现有 `MarketDataService` 能使用模拟 iFinD Provider。

等正式权限到位后，再增加一个默认跳过的真实集成测试。它只在明确设置
专用环境开关和凭证时运行。

## 8. 验收标准

- 无 iFinD 凭证时，QuantLab 可正常启动，AkShare 功能和全部现有测试
  不受影响。
- 模拟 iFinD 凭证和 HTTP 响应时，用户通过现有页面流程获得标准行情，
  且数据被现有缓存和版本流程保存。
- 页面和日志不泄露 token。
- 通过环境配置可切换 `akshare`、`ifind` 和 `auto`，不修改页面、
  分析或回测代码。
- 真实权限到位后，只需在服务器密钥配置中填入 `refresh_token`
  并运行集成测试，无需重写 Provider。

## 9. 已知限制

- 当前无真实权限，只能保证根据公开接口约定和模拟响应完成实现；
- iFinD 实际账号的接口权限、频率、字段和可获取历史范围以合同与
  实际响应为准；
- 多用户间接访问、数据缓存和展示仍需在对外内测前取得相应授权。
