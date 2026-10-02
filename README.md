# RAG 智能客服问答系统（简历版）

一个基于 **RAG（检索增强生成）** 的智能客服系统：把多份简历存入 **Milvus 向量数据库**，
用 **智谱 embedding 做语义向量化**、**DeepSeek 做回答大模型**；客户在输入窗口提问，
如果答案在简历中则精准回答，否则弹框“是否切换到人工服务”，确认后进入人工服务，
消息实时转发到**客服工作台**，客服可回复；客户关闭窗口时，客服端显示“客户已下线”。

---

## 一、功能特性

- ✅ 大模型问答：用 **DeepSeek（deepseek-chat）** 依据简历内容生成自然语言答案，只回答简历里有的、不编造。
- ✅ 语义向量化：用 **智谱（embedding-2，1024 维）** 把简历分块与问题编码为语义向量，存入 Milvus 做精准匹配。
- ✅ 多简历入库：支持一个或多个 PDF 简历，自动解析（含中文 PDF 的 ToUnicode 还原）、分块、向量化写入 Milvus。
- ✅ 忠实问答：只依据简历内容回答；无法回答时给出明确提示。
- ✅ 多轮记忆：同一会话的历史对话存入 Redis，后续提问能理解“他/那”等指代，回答连贯。
- ✅ 转人工：无法回答 → 弹框“我无法回答当前问题，是否切换到人工服务？”→ 点击“是”切换人工。
- ✅ 多客服工作台：每个客服一个独立工作台（各自一个标签页），客户转人工时**按最少负载路由**到对应客服，互不可见。
- ✅ 客服掉线转接：客服关闭工作台后，其名下客户自动转接给其它在线客服；无人在线则客户排队等待。
- ✅ 人工服务：客户提问实时转发到所属客服工作台，客服回复实时回传，**SSE 保持长连接**（无需 websockets 库）。
- ✅ 下线感知：客户关闭窗口，其所属客服端立即显示“客户已下线”。
- ✅ 降级兜底：智谱/DeepSeek 不可用时，自动降级为本地字符哈希向量 + 原文回退，系统仍可运行。

## 二、目录结构

```
rag_customer_service/
├── run.py                 # 启动入口（python run.py）
├── run.bat                # Windows 双击启动
├── config.py              # 全部可配置项（从环境变量/.env 读取）
├── .env                   # 真实密钥（已被 .gitignore 忽略，不提交）
├── .env.example           # 密钥配置模板（提交到版本库）
├── .gitignore             # 忽略 .env / .venv / __pycache__ 等
├── requirements.txt       # 依赖清单
├── core/
│   ├── pdf_extract.py     # 纯 Python PDF 文本提取（支持中文 CID 字体）
│   ├── embedding.py       # 智谱 embedding 向量化（API 不可用时降级哈希）
│   ├── milvus_client.py   # Milvus RESTful API 客户端（无需 pymilvus）
│   ├── bm25.py            # BM25 精确检索 + 意图同义词扩展
│   ├── redis_store.py     # Redis 对话历史存储（极简 RESP 客户端，无需 redis 包）
│   └── rag.py             # 分块入库 / 向量检索 / DeepSeek 回答 / 能否回答判定
├── app/
│   ├── main.py            # FastAPI 后端（REST + SSE 客服会话）
│   └── static/
│       ├── index.html     # 客户窗口
│       └── agent.html     # 客服工作台
├── scripts/
│   ├── ingest.py          # 命令行入库脚本
│   └── check_apis.py      # DeepSeek / 智谱 连通性自检
└── resumes/               # 放更多简历 PDF 的地方（可选）
```

## 三、环境要求

| 依赖 | 说明 |
|---|---|
| Python | 3.9+（已在 3.11 / 3.13 验证） |
| Milvus | 已启动，REST 接口 `http://localhost:19530` |
| Python 包 | `fastapi` `uvicorn` `requests` `numpy`（均已安装） |
| DeepSeek | 回答大模型（`deepseek-chat`），需联网 + API Key |
| 智谱 | 向量化模型（`embedding-2`），需联网 + API Key |
| Redis | 本机 `127.0.0.1:6379`（对话历史存储），密码见 `.env` |

> 无需安装 `pymilvus`、`sentence-transformers`、`pypdf`、`openai`、`websockets`：
> Milvus 走 HTTP REST，向量化/大模型走 HTTP（`requests`），实时通道走 SSE，均无额外依赖。
> DeepSeek/智谱不可用时自动降级为本地哈希向量 + 原文回退，系统仍可运行。

### 密钥配置（.env）

API Key、Redis 密码等**敏感信息不硬编码在代码里**，而是放在项目根目录的 `.env` 文件
（已加入 `.gitignore`，不会提交到版本库）：

```bash
# 首次使用：复制模板并填入你自己的密钥
copy .env.example .env
# 然后编辑 .env，填入 DEEPSEEK_API_KEY / ZHIPU_API_KEY / REDIS_PASSWORD
```

- 代码只通过 `config.py` 读取环境变量（或 `.env`），默认值为空。
- 优先级：系统环境变量 > `.env` > 代码内默认值（仅非敏感项）。
- 未配置密钥时系统会降级运行（本地哈希向量 + 原文回退），并在启动日志里提示。

## 四、快速启动

```bash
cd F:\workplace\rag_customer_service

# （可选）先自检 DeepSeek / 智谱 是否连通
python scripts\check_apis.py

# （可选）命令行预入库
python scripts\ingest.py

# 启动服务
python run.py
```

Windows 下也可直接**双击 `run.bat`**。

| 入口 | 地址 | 用途 |
|---|---|---|
| 客户窗口 | http://127.0.0.1:8000/ | 用户提问 |
| 客服工作台 | http://127.0.0.1:8000/agent | 人工客服接单/回复 |

> 首次启动会自动入库（集合为空时）。默认入库
> `C:\Users\admin\Desktop\袁志杰-Java time.pdf`。
> 启动日志会打印“向量化：智谱 embedding-2 可用，向量维度 1024”等状态。

## 五、使用方式（完整流程）

1. **打开客服工作台（可开多个）**：浏览器访问 `http://127.0.0.1:8000/agent`。
   每个标签页 = 一个独立客服（页头显示工号如“客服-a1b2”）。开多个标签即多名员工。
2. **打开客户窗口**：再开一个窗口访问 `http://127.0.0.1:8000/`。
3. **机器人问答**（DeepSeek 精准回答）：
   - `***会Java吗` → “会，简历中明确写着他精通 JAVA…”
   - `***的电话是多少` → “136********”
   - `他上过什么学校` → “中南民族大学”
   - `求职意向是什么` → “Java”
4. **无法回答时**：例如 `今天天气怎么样`，客户窗口弹出
   **“我无法回答当前问题，是否切换到人工服务？”**。
   - 点 **是**：切换人工，系统按**最少负载**把该客户路由到某个客服工作台，后续消息实时送达该客服。
   - 点 **否**：留在机器人模式继续提问。
5. **人工服务**：被分配到的客服在其工作台卡片回复，客户窗口实时收到；其它客服看不到该客户。
6. **客服掉线转接**：某客服关闭工作台后，其名下客户自动转接给其它在线客服（客户侧提示“已转接给客服xx”）。
7. **客户下线**：关闭客户窗口，其所属客服端立即显示 **“客户已下线”**。

## 六、添加更多简历

1. 把 PDF 放进项目下的 `resumes/` 目录；
2. 或用环境变量追加：`$env:RESUME_PATHS = "D:\简历\张三.pdf;D:\简历\李四.pdf"`；
3. 或改 `config.py` 的 `DEFAULT_RESUME_FILE`，然后重灌：
   ```bash
   curl -X POST http://127.0.0.1:8000/api/ingest
   ```

## 七、配置说明（`config.py` / 环境变量）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | 已填 | 回答大模型 |
| `ZHIPU_API_KEY` / `ZHIPU_EMBED_MODEL` | 已填 | 向量化模型（embedding-2=1024 维 / embedding-3=2048 维） |
| `MILVUS_URI` | `http://localhost:19530` | Milvus 地址 |
| `COLLECTION_NAME` | `resume_kb` | 集合名 |
| `SCORE_THRESHOLD` | `0.40` | 向量余弦阈值：低于此值且 BM25 也低 => 判定无法回答 |
| `BM25_THRESHOLD` | `5.0` | BM25 精确匹配补充判据 |
| `TOP_K` | `5` | 检索返回条数 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `320` / `60` | 分块大小/重叠 |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | `127.0.0.1` / `6379` / 已填 | 历史存储 |
| `REDIS_HISTORY_MAX` / `REDIS_HISTORY_TTL` | `20` / `86400` | 每会话保留条数 / 过期秒数 |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 服务监听地址 |

## 八、技术架构说明

```
客户窗口(index.html) ───SSE+POST──┐
                                  ├─> FastAPI(app/main.py)
客服工作台(agent.html) ─SSE+POST──┘        │
                                           ├─ REST /api/chat /api/ingest /api/stats
                                           │
                     ┌─────────────────────┴──────────────────────┐
                     │           core/rag.py（混合检索）           │
                     │   ① 智谱向量 + Milvus 语义检索（主）         │
                     │   ② BM25 精确匹配（补充判据）                │
                     │   ③ DeepSeek 依据上下文生成答案              │
                     └───────┬───────────────────┬────────────────┘
                             │                   │
                     Milvus 向量库          PDF 解析 + 分块
                (http://localhost:19530)   (core/pdf_extract.py)
```

关键点：

- **向量化**：`core/embedding.py` 调用智谱 `embedding-2` 把简历分块与问题编码为 1024 维
  语义向量写入 Milvus，用余弦相似度做语义匹配；API 不可用时自动降级本地字符哈希。
- **能否回答的判定**：语义向量余弦分（≥ `SCORE_THRESHOLD`）与 BM25 精确匹配分
  （≥ `BM25_THRESHOLD`）二者取并集，兼顾“语义相近”与“关键词精确命中”。
- **答案生成**：把命中的简历片段作为上下文交给 DeepSeek，system 提示“只依据简历如实回答、
  不得编造”，生成自然语言答案；DeepSeek 失败时回退为直接返回命中原文。
- **多轮上下文**：同一 `session_id` 的对话历史存于 Redis（`rag:history:{sid}` 列表，
  键值用 JSON），生成答案时把最近若干轮历史一并传给 DeepSeek，使其能理解“他/那”等
  代词指代、回答连贯；Redis 不可用时自动降级为无历史模式，不影响主流程。
- **中文 PDF**：`core/pdf_extract.py` 解析 FlateDecode 流 + CID 字体 ToUnicode CMap，
  正确还原中文（本简历 PDF 的 ToUnicode 对英文做过移位混淆，已一并还原）。
- **客服会话**：`app/main.py` 用 **SSE（Server-Sent Events）** 维护“客户-客服”实时通道
  （普通 HTTP 流式响应，浏览器原生 `EventSource`，无需 websockets/wsproto 库），
  客户断开连接即广播“客户已下线”通知。

## 九、接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | `{"message":"..."}` → 问答结果 |
| POST | `/api/ingest` | 重新解析并入库全部简历 |
| GET | `/api/stats` | 集合状态/条数 |
| GET | `/api/resumes` | 已配置的简历文件列表 |
| POST | `/api/customer/join` | 客户创建会话，返回 `sid` |
| GET | `/api/customer/stream?sid=` | 客户接收事件的 SSE 流 |
| POST | `/api/customer/chat` | 客户提问 |
| POST | `/api/customer/switch_manual` | 转人工 |
| POST | `/api/customer/end_manual` | 结束人工 |
| GET | `/api/agent/stream` | 客服接收事件的 SSE 流 |
| POST | `/api/agent/reply` | 客服回复客户 |

## 十、常见问题

- **自检 API**：运行 `python scripts\check_apis.py`，分别测试 DeepSeek 与智谱连通性。
- **启动提示连接 Milvus 失败**：确认 Milvus 已启动，浏览器打开 `http://localhost:19530/v1/vector/collections` 应返回 JSON。
- **回答总是“无法回答”**：先 `python scripts\ingest.py` 确认入库条数 > 0；换简历后需重新入库。
- **端口被占用**：`$env:PORT = 8001` 后重启（PyCharm 里记得先停掉旧进程再运行）。
- **多个客户同时接入**：服务端没有并发上限（已实测 12 个客户并发正常）。每个客户窗口占 1 条
  SSE 长连接，同一浏览器对同一站点的 HTTP/1.1 长连接有限制（Chrome/Edge 约 6 条），
  客服工作台也占 1 条，所以同一浏览器里开太多客户标签会“多了没反应”。演示多用客户时：
  ① 每个客户用不同浏览器或无痕窗口（各自独立的连接池）；② 客服工作台单独放另一个浏览器。
  真实部署时每个客户在自己设备上，互不影响。
- **换更强的向量化**：把 `ZHIPU_EMBED_MODEL` 改为 `embedding-3`（2048 维）并重新入库。
