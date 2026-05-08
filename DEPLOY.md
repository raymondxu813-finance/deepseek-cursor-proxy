# 部署指南 (仅限自己使用)

在新电脑上从零部署 deepseek-cursor-proxy。

## 1. 环境准备

```bash
# 确保 Python 3.10+
python3 --version

# 安装 ngrok（用于公网隧道）
brew install ngrok        # macOS

# 配置 ngrok authtoken
ngrok config add-authtoken 3DS6L2EayzvacGPbvWuwTEi4iiW_6paU71De1hzn8Fzg74wqo
```

## 2. 克隆并安装

```bash
git clone <repo-url> deepseek-cursor-proxy
cd deepseek-cursor-proxy

# 安装依赖
pip install -e .
```

## 3. 配置文件

把以下内容写入 ~/.deepseek-cursor-proxy/config.yaml：

```yaml
base_url: https://api.deepseek.com
model: deepseek-v4-pro
thinking: enabled
reasoning_effort: max
display_reasoning: true
collapsible_reasoning: true

host: 127.0.0.1
port: 9000
ngrok: true
verbose: false
request_timeout: 300
max_request_body_bytes: 20971520
cors: false

reasoning_content_path: reasoning_content.sqlite3
missing_reasoning_strategy: recover
reasoning_cache_max_age_seconds: 2592000
reasoning_cache_max_rows: 100000
```

### 配置项说明


| 配置项                             | 当前值                                                  | 说明                             |
| ------------------------------- | ---------------------------------------------------- | ------------------------------ |
| base_url                        | [https://api.deepseek.com](https://api.deepseek.com) | DeepSeek API 地址                |
| model                           | deepseek-v4-pro                                      | 默认模型                           |
| thinking                        | enabled                                              | thinking 模式 (enabled/disabled) |
| reasoning_effort                | max                                                  | 推理力度 (max/medium/low)          |
| display_reasoning               | true                                                 | 在 Cursor 中显示推理过程               |
| collapsible_reasoning           | true                                                 | 推理内容可折叠                        |
| host                            | 127.0.0.1                                            | 本地监听地址                         |
| port                            | 9000                                                 | 本地监听端口                         |
| ngrok                           | true                                                 | 启用 ngrok 公网隧道                  |
| verbose                         | false                                                | 详细日志                           |
| request_timeout                 | 300                                                  | 上游超时（秒）                        |
| max_request_body_bytes          | 20971520                                             | 最大请求体（20MB）                    |
| cors                            | false                                                | CORS 开关                        |
| reasoning_content_path          | reasoning_content.sqlite3                            | 缓存数据库路径                        |
| missing_reasoning_strategy      | recover                                              | 缺失策略 (recover/reject)          |
| reasoning_cache_max_age_seconds | 2592000                                              | 缓存有效期（30天）                     |
| reasoning_cache_max_rows        | 100000                                               | 缓存最大行数                         |


## 4. 启动服务

### 方式一：前台运行（调试用）

```bash
cd ~/deepseek-cursor-proxy
deepseek-cursor-proxy
```

输出示例：

```
default_model: deepseek-v4-pro (thinking, max)
local_base_url: http://127.0.0.1:9000/v1
api_base_url: https://xxxx-xxx.ngrok-free.app/v1
```

### 方式二：后台运行（日常使用）

```bash
cd ~/deepseek-cursor-proxy
nohup deepseek-cursor-proxy > ~/.deepseek-cursor-proxy/proxy.log 2>&1 &
```

### 方式三：screen（推荐，可随时看日志）

```bash
screen -S proxy
cd ~/deepseek-cursor-proxy
deepseek-cursor-proxy
# Ctrl+A D 分离，screen -r proxy 重新连接
```

### 停止服务

```bash
ps aux | grep deepseek-cursor-proxy
kill <PID>
```

## 5. 配置 Cursor

### 步骤 1：获取 API Base URL

启动服务后，看终端输出的 api_base_url，例如：

```
api_base_url: https://xxxx-xxx.ngrok-free.app/v1
```

> 不用 ngrok 时 URL 为 [http://127.0.0.1:9000/v1](http://127.0.0.1:9000/v1)

### 步骤 2：Cursor Settings

1. 打开 Cursor → Settings (Cmd+,)
2. 左侧找到 Models
3. 找到 OpenAI API Key 配置区域：
  - API Key：填写你的 DeepSeek API Key（sk-xxxx）
  - Base URL：填写 api_base_url 的值（必须以 /v1 结尾）
4. 在 Models 列表确保 deepseek-v4-pro 已启用

### 步骤 3：验证

Cursor 中新建 Chat，选择 deepseek-v4-pro，发送消息。终端出现 request model=deepseek-v4-pro 即成功。

## 6. 日常流程

```bash
# 开机后
screen -S proxy
cd ~/deepseek-cursor-proxy && deepseek-cursor-proxy

# 然后打开 Cursor，模型选 deepseek-v4-pro 即可
```

ngrok 断开时重启 proxy 即可（URL 可能变化，需更新 Cursor Base URL）。

## 7. 常见问题

**看日志：** tail -f ~/.deepseek-cursor-proxy/proxy.log

**详细日志：** deepseek-cursor-proxy --verbose

**本地测试：** config.yaml 中 ngrok: false，Cursor Base URL 填 [http://127.0.0.1:9000/v1](http://127.0.0.1:9000/v1)

## 8. 迁移缓存（可选）

```bash
scp ~/.deepseek-cursor-proxy/reasoning_content.sqlite3 user@new:~/.deepseek-cursor-proxy/
```

## 9. ngrok authtoken

```
3DS6L2EayzvacGPbvWuwTEi4iiW_6paU71De1hzn8Fzg74wqo
```

