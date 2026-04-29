# Week 3 - Custom MCP Server

一个自定义的 MCP (Model Context Protocol) 服务器实现，封装了天气和新闻 API，支持与 Claude Desktop 等 MCP 客户端集成。

## 项目结构

```
week3/
├── src/
│   ├── __init__.py
│   ├── main.py              # MCP 服务器主入口
│   ├── config.py            # 配置管理
│   ├── logger.py            # 日志工具（使用 stderr）
│   ├── tools/               # Tools 层
│   │   ├── __init__.py
│   │   ├── base.py          # 工具基类和注册器
│   │   ├── weather.py       # 天气工具
│   │   └── news.py          # 新闻工具
│   └── capabilities/        # 能力层
│       ├── __init__.py
│       ├── tool_manager.py  # 工具管理器
│       ├── resource_manager.py  # 资源管理器
│       ├── prompt_manager.py    # 提示模板管理器
│       └── capability_layer.py  # 能力层主控制器
├── tests/
│   ├── __init__.py
│   ├── test_tools.py        # Tools 层单元测试
│   └── test_capabilities.py # 能力层单元测试
├── .env.example             # 环境变量示例
├── requirements.txt         # Python 依赖
├── README.md                # 本文档
└── assignment.md            # 作业要求
```

## 架构设计

### 分层架构

本项目采用清晰的分层架构设计：

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Client (Claude Desktop)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (main.py)                      │
│  - 协议通信 (JSON-RPC 2.0)                                   │
│  - 传输层 (STDIO)                                            │
│  - 消息路由                                                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Capability Layer (能力层)                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ ToolManager  │  │ResourceManager│ │PromptManager │    │
│  │              │  │              │  │              │    │
│  │ - 自动注册   │  │ - 静态资源   │  │ - 模板管理   │    │
│  │ - 动态发现   │  │ - 动态资源   │  │ - 参数渲染   │    │
│  │ - 工具执行   │  │ - 订阅通知   │  │ - 自动发现   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tools Layer (工具层)                       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  BaseTool    │  │ WeatherTool  │  │  NewsTool    │    │
│  │              │  │              │  │              │    │
│  │ - 抽象基类   │  │ - 天气 API   │  │ - 新闻 API   │    │
│  │ - 接口定义   │  │ - 错误处理   │  │ - 类别筛选   │    │
│  │ - 类型检查   │  │ - 速率限制   │  │ - 分页支持   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件说明

#### Tools Layer (工具层)

**职责**：定义具体的工具实现，封装外部 API 调用

**核心类**：
- `BaseTool`：工具抽象基类，定义了工具的基本接口
  - `name`：工具名称
  - `description`：工具描述
  - `get_input_schema()`：返回输入参数的 JSON Schema
  - `execute(**kwargs)`：异步执行工具逻辑
- `ToolRegistry`：工具注册器（单例模式）
  - `register(tool)`：注册工具
  - `unregister(name)`：取消注册
  - `get(name)`：获取工具
  - `list_definitions()`：列出所有工具定义

**已实现的工具**：
1. `WeatherTool`：天气查询工具
   - 封装 OpenWeatherMap API
   - 支持城市名称查询
   - 支持摄氏度/华氏度切换
   - 错误处理和速率限制

2. `NewsTool`：新闻搜索工具
   - 封装 NewsAPI
   - 支持关键词搜索
   - 支持类别筛选（business, technology, sports 等）
   - 支持多语言
   - 支持分页和排序

#### Capability Layer (能力层)

**职责**：管理工具、资源和提示模板，提供统一的能力接口

**核心类**：
- `ToolManager`：工具管理器
  - `auto_register_tools()`：自动扫描并注册工具类
  - `register_tool(tool)`：手动注册工具
  - `execute_tool(name, **kwargs)`：执行工具
  - `list_tools()`：列出所有工具定义

- `ResourceManager`：资源管理器
  - `register_static_resource()`：注册静态资源
  - `register_dynamic_resource()`：注册动态资源（带生成器）
  - `register_file_resource()`：从文件注册资源
  - `subscribe(uri, callback)`：订阅资源更新
  - `notify_subscribers(uri)`：通知订阅者

- `PromptManager`：提示模板管理器
  - `register_prompt()`：注册提示模板
  - `register_prompt_from_file()`：从文件注册
  - `auto_register_prompts()`：自动扫描并注册
  - `render_prompt(name, **kwargs)`：渲染提示模板

- `CapabilityLayer`：能力层主控制器
  - `initialize(auto_register=True)`：初始化能力层
  - `get_capabilities()`：获取能力摘要
  - `list_tools()`、`execute_tool()`：工具相关操作
  - `list_resources()`、`get_resource()`：资源相关操作
  - `list_prompts()`、`render_prompt()`：提示相关操作
  - `shutdown()`：关闭能力层

## 安装配置

### 前置条件

- Python 3.10+
- pip 或 poetry

### 安装依赖

```bash
cd week3
pip install -r requirements.txt
```

### 环境配置

1. 复制环境变量示例文件：

```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API 密钥：

```env
# OpenWeatherMap API Key (免费注册: https://openweathermap.org/api)
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here

# NewsAPI Key (免费注册: https://newsapi.org/)
NEWSAPI_KEY=your_newsapi_key_here
```

### 获取 API 密钥

#### OpenWeatherMap API Key

1. 访问 https://openweathermap.org/api
2. 注册账号
3. 验证邮箱
4. 在个人设置中获取 API Key
5. 免费 tier 每分钟 60 次请求，每天 1000 次

#### NewsAPI Key

1. 访问 https://newsapi.org/
2. 点击 "Get API Key"
3. 注册账号
4. 获取 API Key
5. 免费 tier 支持开发者使用，有速率限制

## 使用方法

### 与 Claude Desktop 集成

1. 找到 Claude Desktop 的配置文件：

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. 编辑配置文件，添加 MCP 服务器：

```json
{
  "mcpServers": {
    "custom-mcp-server": {
      "command": "python",
      "args": [
        "-m",
        "src.main"
      ],
      "env": {
        "OPENWEATHERMAP_API_KEY": "your_api_key_here",
        "NEWSAPI_KEY": "your_api_key_here"
      },
      "cwd": "/absolute/path/to/week3"
    }
  }
}
```

**注意**：
- 将 `cwd` 替换为你的 week3 目录的绝对路径
- 或者在 `.env` 文件中设置环境变量，就不需要在配置中重复设置

3. 重启 Claude Desktop

4. 现在你可以在 Claude Desktop 中使用这些工具了！

### 示例对话

#### 天气查询

```
用户: 北京今天天气怎么样？

Claude: 我来查询一下北京的天气...

[调用 get_weather 工具，参数: city="Beijing", units="metric"]

工具返回:
{
  "success": true,
  "city": "Beijing",
  "country": "CN",
  "temperature": {
    "current": "25.5°C",
    "feels_like": "26.0°C"
  },
  "weather": {
    "main": "Clear",
    "description": "clear sky"
  },
  "humidity": "50%",
  "wind": {
    "speed": "2.5 m/s"
  }
}

Claude: 北京今天天气晴朗，气温约 25.5°C，体感温度 26°C，湿度 50%，风速 2.5 m/s。非常适合户外活动！
```

#### 新闻搜索

```
用户: 帮我找一些关于人工智能的最新新闻

Claude: 我来搜索一下关于人工智能的最新新闻...

[调用 search_news 工具，参数: query="人工智能", language="zh", sort_by="publishedAt", page_size=5]

工具返回:
{
  "success": true,
  "total_results": 1234,
  "articles_count": 5,
  "articles": [
    {
      "title": "GPT-5 传闻：OpenAI 正在开发下一代大语言模型",
      "description": "据知情人士透露，OpenAI 已经开始研发 GPT-5...",
      "url": "https://example.com/news/gpt5",
      "source": {"name": "Tech News"},
      "published_at": "2024-01-15T10:30:00Z"
    },
    ...
  ]
}

Claude: 以下是关于人工智能的最新新闻：

1. **GPT-5 传闻：OpenAI 正在开发下一代大语言模型**
   - 来源：Tech News
   - 发布时间：2024-01-15 10:30
   - 据知情人士透露，OpenAI 已经开始研发 GPT-5...
   - [阅读更多](https://example.com/news/gpt5)

2. ...
```

### 动态注册工具

你可以在运行时动态注册新的工具：

```python
from src.capabilities import CapabilityLayer
from src.tools.base import BaseTool
from typing import Any

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_custom_tool",
            description="这是一个自定义工具"
        )
    
    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "参数描述"
                }
            },
            "required": ["param"]
        }
    
    async def execute(self, param: str) -> dict[str, Any]:
        return {"success": True, "result": f"处理: {param}"}

# 使用能力层注册
cap_layer = CapabilityLayer()
cap_layer.initialize()
cap_layer.register_tool(MyCustomTool())
```

## 工具参考

### get_weather

**描述**：获取指定城市的当前天气信息

**参数**：
| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| city | string | 是 | 城市名称（英文，如 Beijing, Tokyo, London） |
| units | string | 否 | 温度单位，可选值: `metric`（摄氏度）、`imperial`（华氏度），默认 `metric` |

**返回示例**：
```json
{
  "success": true,
  "city": "Beijing",
  "country": "CN",
  "coordinates": {
    "latitude": 39.9,
    "longitude": 116.4
  },
  "weather": {
    "main": "Clear",
    "description": "clear sky",
    "icon": "01d"
  },
  "temperature": {
    "current": "25.5°C",
    "feels_like": "26.0°C",
    "min": "24.0°C",
    "max": "27.0°C"
  },
  "humidity": "50%",
  "pressure": "1013 hPa",
  "wind": {
    "speed": "2.5 m/s",
    "direction": "180°"
  },
  "visibility": "10 km",
  "clouds": "0%",
  "sunrise": 1234567890,
  "sunset": 1234567890
}
```

### search_news

**描述**：搜索最新新闻文章

**参数**：
| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| query | string | 否 | 搜索关键词或短语（与 category 二选一） |
| category | string | 否 | 新闻类别（与 query 二选一） |
| language | string | 否 | 语言代码，默认 `en` |
| sort_by | string | 否 | 排序方式，默认 `publishedAt` |
| page_size | integer | 否 | 返回数量，1-100，默认 20 |
| page | integer | 否 | 页码，默认 1 |

**category 可选值**：
- `business` - 商业
- `entertainment` - 娱乐
- `general` - 综合
- `health` - 健康
- `science` - 科学
- `sports` - 体育
- `technology` - 科技

**language 可选值**：
- `ar` - 阿拉伯语
- `de` - 德语
- `en` - 英语
- `es` - 西班牙语
- `fr` - 法语
- `he` - 希伯来语
- `it` - 意大利语
- `nl` - 荷兰语
- `no` - 挪威语
- `pt` - 葡萄牙语
- `ru` - 俄语
- `sv` - 瑞典语
- `zh` - 中文

**sort_by 可选值**：
- `relevancy` - 按相关性
- `popularity` - 按流行度
- `publishedAt` - 按发布时间（最新）

**返回示例**：
```json
{
  "success": true,
  "total_results": 1234,
  "page": 1,
  "page_size": 20,
  "articles_count": 20,
  "articles": [
    {
      "title": "新闻标题",
      "description": "新闻描述",
      "content": "新闻内容",
      "url": "https://example.com/news",
      "url_to_image": "https://example.com/image.jpg",
      "source": {
        "id": "source-id",
        "name": "来源名称"
      },
      "author": "作者",
      "published_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

## 运行测试

本项目包含完整的单元测试，使用 pytest 和 pytest-asyncio。

### 运行所有测试

```bash
cd week3
pytest tests/ -v
```

### 运行特定测试

```bash
# 只运行 Tools 层测试
pytest tests/test_tools.py -v

# 只运行能力层测试
pytest tests/test_capabilities.py -v

# 运行特定测试函数
pytest tests/test_tools.py::TestWeatherTool::test_execute_success -v
```

### 测试覆盖率报告

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### 测试说明

#### Tools 层测试 (`tests/test_tools.py`)

测试内容：
- `TestBaseTool`：测试基类的初始化、获取定义、执行功能
- `TestToolRegistry`：测试单例模式、注册、重复注册、取消注册、列出定义
- `TestWeatherTool`：测试天气工具的输入验证、错误处理、成功执行（使用 mock）
- `TestNewsTool`：测试新闻工具的输入验证、错误处理、成功执行（使用 mock）

**Mock 策略**：
- 使用 `unittest.mock.AsyncMock` 模拟 HTTP 客户端
- 避免真实 API 调用，确保测试可重复执行
- 模拟各种错误场景（404、401、超时等）

#### 能力层测试 (`tests/test_capabilities.py`)

测试内容：
- `TestToolManager`：测试工具管理器的所有功能
- `TestResourceManager`：测试静态资源、动态资源、文件资源、订阅通知
- `TestPromptManager`：测试提示模板的注册、渲染、自动发现
- `TestCapabilityLayer`：测试能力层的整合功能

**测试特点**：
- 使用 `tempfile` 创建临时文件测试文件资源
- 测试异步生成器的动态资源
- 测试完整的初始化和关闭流程

## 错误处理

### 工具执行错误

所有工具都返回统一的错误格式：

```json
{
  "success": false,
  "error": "错误描述信息"
}
```

### 常见错误类型

| 错误类型 | 描述 | 处理方式 |
|----------|------|----------|
| API key 未配置 | 环境变量未设置 | 返回友好提示，引导用户配置 |
| 城市/查询为空 | 必填参数缺失 | 返回参数验证错误 |
| 404 Not Found | 资源不存在 | 返回特定错误信息 |
| 401 Unauthorized | API key 无效 | 提示检查 API key |
| 429 Too Many Requests | 超过速率限制 | 提示稍后重试 |
| 超时 | 网络请求超时 | 提示检查网络连接 |
| 连接错误 | 无法连接到 API | 提示检查网络 |

### 速率限制

工具会检测 API 的速率限制响应头：
- `X-RateLimit-Remaining`：剩余请求数
- `X-RateLimit-Reset`：重置时间戳

当检测到速率限制时，会返回友好的错误提示。

## 日志记录

本项目使用 `stderr` 进行日志记录（符合 MCP STDIO 传输要求）。

### 日志级别

- `DEBUG`：详细的调试信息
- `INFO`：一般信息（工具调用、初始化等）
- `WARNING`：警告信息
- `ERROR`：错误信息
- `CRITICAL`：严重错误

### 日志格式

```
2024-01-15 10:30:00 - mcp-server - INFO - Calling tool: get_weather with arguments: {'city': 'Beijing'}
```

### 配置日志级别

在 `.env` 文件中设置：

```env
LOG_LEVEL=DEBUG  # 可选: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## 扩展开发

### 添加新工具

1. 创建新的工具类，继承 `BaseTool`：

```python
from src.tools.base import BaseTool
from typing import Any

class MyNewTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_new_tool",
            description="工具描述"
        )
    
    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "参数描述"
                }
            },
            "required": ["param1"]
        }
    
    async def execute(self, param1: str) -> dict[str, Any]:
        try:
            # 执行工具逻辑
            result = await self._do_something(param1)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _do_something(self, param: str) -> Any:
        # 实际的实现逻辑
        pass
```

2. 在 `src/tools/__init__.py` 中导出：

```python
from .my_new_tool import MyNewTool

__all__ = [
    # ... 现有的导出
    "MyNewTool"
]
```

3. 工具会被 `ToolManager.auto_register_tools()` 自动发现并注册，或者手动注册：

```python
from src.capabilities import CapabilityLayer
from src.tools import MyNewTool

cap_layer = CapabilityLayer()
cap_layer.initialize(auto_register=False)
cap_layer.register_tool(MyNewTool())
```

### 添加资源

```python
from src.capabilities import CapabilityLayer

cap_layer = CapabilityLayer()
cap_layer.initialize()

# 注册静态资源
cap_layer.resource_manager.register_static_resource(
    uri="config://settings",
    name="应用设置",
    description="应用程序配置",
    content={"theme": "dark", "language": "zh"},
    mime_type="application/json"
)

# 注册动态资源
async def get_server_status():
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

cap_layer.resource_manager.register_dynamic_resource(
    uri="status://server",
    name="服务器状态",
    description="实时服务器状态",
    generator=get_server_status,
    mime_type="application/json"
)
```

### 添加提示模板

```python
from src.capabilities import CapabilityLayer
from src.capabilities.prompt_manager import PromptArgument

cap_layer = CapabilityLayer()
cap_layer.initialize()

cap_layer.prompt_manager.register_prompt(
    name="code_review",
    description="代码审查提示模板",
    template="""请审查以下代码：

代码：
{code}

请从以下方面进行审查：
1. 代码正确性
2. 性能优化
3. 安全问题
4. 代码风格

语言：{language}""",
    arguments=[
        PromptArgument(name="code", description="要审查的代码", required=True),
        PromptArgument(name="language", description="回复语言", required=False, default="中文")
    ]
)

# 使用模板
prompt = cap_layer.render_prompt("code_review", code="print('hello')", language="中文")
```

## 部署选项

### 本地 STDIO 模式（默认）

这是最简单的部署方式，适用于 Claude Desktop 等本地 MCP 客户端。

**配置方式**：参见「与 Claude Desktop 集成」章节

### 远程 HTTP 模式（额外学分）

可以使用 SSE (Server-Sent Events) 传输方式部署为远程 HTTP 服务器。

**注意**：当前 `main.py` 只实现了 STDIO 模式。要实现 HTTP 模式，需要：

1. 使用 `mcp.server.sse` 模块
2. 处理 HTTP 认证（API Key 或 OAuth2）
3. 部署到云服务（Cloudflare Workers、Vercel 等）

如果你需要实现远程 HTTP 模式，请告诉我，我可以帮你扩展。

## 安全考虑

### API 密钥管理

- 不要将 API 密钥提交到版本控制
- 使用 `.env` 文件或环境变量
- `.gitignore` 已包含 `.env` 文件

### 工具执行安全

- 所有工具调用都会被 MCP 客户端（如 Claude Desktop）记录
- 用户可以在调用前查看和确认工具参数
- 建议在生产环境中添加额外的权限检查

### 速率限制

- 工具会尊重 API 的速率限制
- 当检测到速率限制时，会返回友好的错误提示
- 建议在生产环境中添加缓存机制

## 故障排查

### 常见问题

**1. Claude Desktop 无法连接到 MCP 服务器**

检查：
- 配置文件路径是否正确
- `cwd` 是否指向 week3 目录的绝对路径
- Python 环境是否正确
- 依赖是否已安装

**2. 工具返回 API key 错误**

检查：
- `.env` 文件是否存在
- API key 是否正确
- 环境变量名称是否正确（`OPENWEATHERMAP_API_KEY`、`NEWSAPI_KEY`）

**3. 测试失败**

检查：
- 测试文件是否在 `tests/` 目录
- 依赖是否已安装（包括 `pytest` 和 `pytest-asyncio`）
- Python 版本是否为 3.10+

**4. 日志不显示**

检查：
- `LOG_LEVEL` 环境变量设置
- MCP 服务器使用 `stderr` 输出日志，不是 `stdout`

### 调试技巧

**1. 手动测试 MCP 服务器**

虽然 MCP 服务器使用 STDIO 通信，但你可以通过测试来验证：

```bash
cd week3
pytest tests/ -v
```

**2. 检查 API 密钥是否有效**

可以使用 curl 手动测试 API：

```bash
# 测试 OpenWeatherMap
curl "https://api.openweathermap.org/data/2.5/weather?q=Beijing&appid=YOUR_API_KEY"

# 测试 NewsAPI
curl "https://newsapi.org/v2/top-headlines?q=technology&apiKey=YOUR_API_KEY"
```

## 作业要求对照

| 要求 | 实现状态 | 说明 |
|------|----------|------|
| 选择外部 API | ✅ 完成 | OpenWeatherMap（天气）、NewsAPI（新闻） |
| 至少 2 个 MCP 工具 | ✅ 完成 | `get_weather`、`search_news` |
| 错误处理 | ✅ 完成 | HTTP 错误、超时、空结果、速率限制 |
| 速率限制 | ✅ 完成 | 检测 API 速率限制响应头 |
| STDIO 传输 | ✅ 完成 | 支持与 Claude Desktop 集成 |
| 清晰的文档 | ✅ 完成 | 本文档包含详细说明 |
| 工具自动注册 | ✅ 完成 | `ToolManager.auto_register_tools()` |
| 资源管理 | ✅ 完成 | `ResourceManager` 支持静态/动态资源 |
| 提示模板管理 | ✅ 完成 | `PromptManager` 支持模板注册和渲染 |
| 单元测试 | ✅ 完成 | `tests/` 目录包含完整测试 |

## 许可证

本项目是课程作业的一部分，仅供学习使用。

## 联系方式

如有问题，请查看代码注释或参考以下资源：
- [MCP 官方文档](https://modelcontextprotocol.io/)
- [OpenWeatherMap API 文档](https://openweathermap.org/api)
- [NewsAPI 文档](https://newsapi.org/docs)
