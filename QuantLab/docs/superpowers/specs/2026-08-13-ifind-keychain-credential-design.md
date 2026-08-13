# iFinD 凭证钥匙串持久化设计

## 目标

在 macOS 本地安全持久化 iFinD `refresh_token`，使 QuantLab 重启终端或电脑后仍可以连接 iFinD，同时确保真实凭证不进入源代码、`.env` 文件、日志、测试输出或 Git 仓库。

## 范围

本次只实现 macOS 本地凭证管理与安全启动。不修改 iFinD Provider、行情服务、分析和回测模块；不实现网站服务器的秘密管理，不向前端暴露凭证。

## 保存方式

- 使用 macOS Keychain 保存一条 generic password。
- service 名固定为 `QuantLab-iFinD`。
- account 名固定为当前 macOS 用户名。
- 配置命令通过隐藏输入读取 token，不把 token 放在命令行参数中。
- 重复配置时更新同一条钥匙串记录，不创建多份凭证。

## 用户命令

新增两个项目脚本：

1. `./configure_ifind_mac.command`
   - 确认当前系统是 macOS；
   - 隐藏输入 `refresh_token`；
   - 拒绝空值；
   - 写入 macOS Keychain；
   - 只输出“已安全保存”，不输出 token。

2. `./run_ifind_mac.command`
   - 从 macOS Keychain 读取 token；
   - 若不存在，用自然语言提示先运行配置脚本并终止；
   - 仅对 Streamlit 子进程设置 `IFIND_REFRESH_TOKEN` 和 `MARKET_DATA_PROVIDER=ifind`；
   - 激活项目虚拟环境并以 `PYTHONPATH=src` 启动 `app.py`；
   - 退出时不将 token 写入 shell 配置或项目文件。

## 数据流

```text
用户隐藏输入 token
→ macOS Keychain
→ 启动脚本在运行时读取
→ Streamlit 子进程环境变量
→ Provider Factory
→ IFindQuantAPIClient
```

token 不会进入 Streamlit 页面、行情缓存元数据或 QuantLab 日志。

## 错误处理

- 非 macOS：提示该脚本仅适用于 macOS。
- 未安装或损坏虚拟环境：提示重建 `.venv`。
- 钥匙串中无 token：提示运行配置脚本。
- 钥匙串访问失败：不回显 token，仅提示检查钥匙串权限。
- iFinD 认证失败：由现有 Provider 错误处理向页面返回友好提示。

## 测试与验收

- 脚本语法检查通过。
- 使用临时替身 `security` 命令测试配置和启动流程，测试不写入真实钥匙串，不发起网络请求。
- 测试凭证缺失时会拒绝启动。
- 测试启动时会选择 iFinD，但脚本输出不包含测试 token。
- 完整 Python 离线测试仍需通过。
- 人工验收：关闭原终端后，使用新脚本启动 QuantLab，页面仍能通过 iFinD 获取一个 ETF 的行情。

## 未来网站部署

公开网站不使用 macOS Keychain。部署时由服务器或云平台 Secret Manager 注入同名环境变量 `IFIND_REFRESH_TOKEN`，Provider 和页面不需修改。多用户使用前仍需确认 iFinD 授权允许服务端间接访问、数据展示与缓存。
