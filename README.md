# ragbot — 能上线的多模态 RAG 问答服务

文档入库（PDF / Word 正文 / 表格 / 图片三模态）→ 混合检索（稠密向量 + BM25 两路召回、RRF 融合）→ 流式生成。**Web 聊天（SSE 打字机）+ 钉钉机器人（Stream 模式免公网）双客户端**，带完整的生产化横切面：评测集、请求级追踪、超时/有限重试/降级、限流与 token 预算、压测数据。

> 源于教学项目 [Mini-RAG](https://github.com/alicexl/Mini-RAG)（三步从零搭 RAG 的原理线）；本仓是其原 step4 应用的产品线独立演化——检索/解析内核在 Mini-RAG 内完成 E2E 与评测验证后迁出，围绕「能上线」补齐工程横切面。

## 架构

### 总览

```
                    ┌─ Web 客户端（app/static，SSE 流式 + 文档上传）
请求 → Flask 路由 ──┤
                    └─ 钉钉机器人（app/bot.py，Stream 出站 WebSocket 免公网，
                       AI 卡片打字机，会话与使用日志服务端持有）
        │
   pipeline（app/pipeline.py：单例接线，web / bot / eval 三个调用方
            共享同一条链——评测测的就是线上这条链）
        ├─ ingest   PDF / docx 多模态解析 → Chunk → embedding → Milvus
        ├─ retrieve pre-retrieval（PRE_MODE，默认 off）→ embedding
        │           → Milvus 两路召回 + RRF 融合 → Top-5（rerank 已退役）
        └─ generate 拼 context（编号+类型+来源标注）→ 流式生成
        │
   横切面：trace（request_id + span 落 SQLite）· limits（按人日限 + token 日预算）
```

### 代码地图

| 模块 | 职责 |
|---|---|
| `app/main.py` | Flask 入口：路由 + SSE 流式 + 钉钉线程接线 |
| `app/bot.py` | 钉钉 Stream 机器人：会话历史 + 使用日志（一表两用）+ AI 卡片打字机 |
| `app/pipeline.py` | 胶合层：单例接线（init）+ ingest / retrieve / generate 入口 + context 拼装 |
| `app/store.py` | 存储与解析：Chunk 模型、Milvus Lite 封装、PDF/docx 多模态解析、递归切块器 |
| `app/llm.py` | 模型客户端：embedder / rerank / VLM / 流式生成 / pre-retrieval，含重试与超时策略 |
| `app/trace.py` | request_id 贯穿全链 + span 落 SQLite |
| `app/limits.py` | 按人每日问数 + 全局 token 日预算 |
| `eval/` | 评测集（evalset.py）+ 判分 runner（run_eval.py） |
| `loadtest/` | 压测 |

### 入口层：Web + 钉钉，一条链两个壳

Web 端点（`app/main.py`）：

| 端点 | 作用 |
|---|---|
| `GET /` | 聊天页（app/static/index.html） |
| `GET /api/status` | 库状态：chunk 数 + 已导入文档清单 |
| `POST /api/upload` | multipart 上传：.pdf/.docx 走多模态管线，.doc 明确拒绝并引导转存，其余按纯文本；幂等（重复导入跳过） |
| `POST /api/chat` | SSE 流式：先推 `sources`（命中清单：分数/类型/页码/章节/图片链接），再 `delta…` 逐字，末帧 `done` / `error` |
| `POST /api/clear` | drop collection 重建（比删 .db 干净，避开文件锁） |
| `GET /api/media/<名>` | 图片 chunk 原图回显（受控目录，无遍历风险） |

钉钉机器人（`app/bot.py`，`DINGTALK_CLIENT_ID/SECRET` 两 env 设了即随服务起线程）：

- **Stream 模式 = 出站 WebSocket**：本进程主动连钉钉服务器，零公网 IP / 域名 / 证书；未认证个人团队即可用（平台侧开 `Card.Instance.Write` + `Card.Streaming.Write` 两权限）
- **会话状态服务端持有**（Web 端由前端上送，这里 SQLite 按 conversation_id，最近 6 条 = 3 轮）；user 行只存原始问题不带参考资料，防多轮滚雪球
- **history 表同时是使用日志**（问题/答案/来源/延迟/token）——坏例排查与真实 query 积累从这里长出来
- 回复链路：收到消息**先 ACK 再异步回答**（生成长尾会触发钉钉超时重推，message_id 去重）→ 先发 AI 卡片「执行中」占位 → 流式写入（打字机三要素：首帧置态 / 底层帧 / isFinalize 终帧）→ ai_finish 定稿
- 与 Web 共库**同进程**（Milvus 单进程文件锁决定）

### 数据模型：Chunk 一处定义，全链共用

```python
@dataclass Chunk:        # app/store.py —— 入库 / 检索 / 展示共用这一个定义
    text: str            # 可检索文本：embedding 与 BM25 的输入
    chunk_type: str      # text | table | image
    content: str         # 命中时回填给 LLM 的原文（表格=HTML；正文与 text 同）
    page: int            # PDF 页码；非 PDF / 跨页正文 = 0
    section: str = ""    # 章节名（独立表格块用；正文跨章节不硬造）
    source: str = ""     # 来源文件名 stem（幂等键 + 展示 + 归因锚）
```

- **text 与 content 分离**是三模态统一的关键：检索吃文本化版本，生成吃原文（表格 HTML 保行列结构，LLM 按单元格读数远好于线性化文本）
- **独立表格块的 text 带 `source：` 锚前缀**——多模态块脱离正文流独立成块后没有公司名，不带锚则 BM25/dense 两路都吃不到（M05 实证：命中 57.9→84.2%，见演进记录）
- 检索命中 = `RetrievedChunk(Chunk) + rrf/score`，web / bot / eval 三个消费方共用

### 入库（ingest）：三模态进同一检索空间

**PDF（PyMuPDF）**——每页抽三类元素（文本块 / 表格 / 图片）按 y 坐标排序还原「元素流」：

- 页眉（y<80pt）/ 页脚剔除；表格区内的文本块剔除（表格整体转 HTML，不重复计字）；块内硬换行是排版折行，拼回连续文本
- 章节标题（中文序号「一、/（一）」或数字序号「1. / 2.1」正则 + 行短）更新当前章节，记进独立块的 `section`
- **小表格（≤10 行）**：以〔表格Tn〕边界标记跟正文一起切块（上下文不丢），content 版替换回 HTML
- **大表格**：独立成 table 块（text 带 source 锚），正文留引导句
- **图片**：装饰过滤（最短边<50pt / 面积>80% 页面 / 同位叠放 / 跨页同图 md5 去重）→ 原件落 `data/media/` → glm-4v-flash 生成描述（「图表类型+主体+趋势，不推测数字」），描述独立成块参与检索，命中时前端经 `/api/media/` 回显原图

**docx（python-docx）**——与 PDF 同构的三类 chunk：按 body 元素流迭代保持段/表交错序（paragraphs/tables 两个平铺列表会丢阅读顺序）；章节识别两路 = Heading 样式优先 + **加粗+序号正则兜底**（真实世界 docx 常用加粗伪装标题，反样式样本专验这条路）；docx 无页码，page=0

**纯文本**——直接切块。切块器 RecursiveTextSplitter：chunk_size=400 / overlap=50，分隔符优先级 段落→换行→句读→空格→字符级兜底。

入库幂等：source 已在库（标量过滤查询）即跳过。

### 存储与检索：Milvus Lite 库内混合检索

- **单文件库**：生产 `data/mini_rag.db`、评测 `data/eval.db`。单进程文件锁——同一 data 目录同一时刻只允许一个进程打开；web/bot 共库同进程，评测独立库不碰生产、无需停服务
- **一张 collection 装下一切**：`text`（VARCHAR 65535，jieba analyzer——中文按词切不按字切）→ 库内 **BM25 Function** 自动出 sparse 向量（不用手填）；`vector`（1024 维，HNSW/COSINE）；`source/chunk_type/content/page/section` 标量字段随行存
- **检索一次 `hybrid_search` 库内完成**：dense 路（查询向量，COSINE）+ sparse 路（查询原文进来，库内 jieba 分词打 BM25）各捞 RECALL_K=20 → **RRFRanker(k=60) 融合** → 取 Top-K=5 进 context
- rerank 层已退役（智谱分数饱和，见可靠性设计表）；pre-retrieval（rewrite/hyde/multi）默认 off（A/B 结论见评测）

### 生成：接地 + 归因 + 流式

- **context 拼装**：`[n][类型｜来源：source] + content` 逐条编号，模型引用「根据[1]」；来源标注让模型在多文档下按公司/年份正确归因（无标注时实测把盐湖数字安到云天化头上，答对 34.9→52.9%）
- **SYSTEM_PROMPT 四条硬约束**：只据参考资料答 / 用到哪条标编号 / 按公司与年份归因 / 无信息则「资料中未提及」禁编造——评测的拒绝率与归因正确性就是对它的直接检验
- **多轮**：history 原样拼在本轮之前，本轮问题才带参考资料
- **StreamedAnswer**（Web SSE 与钉钉打字机共用同一段代码）：首字/流间隔 15s 超时 → 取消重试一次（长尾多为上游排队抖动，重试大概率落快通道）→ 总时长 90s 兜底上抛 → 关思考（数值问答不需要，首字快 + 省 token）→ usage 从流的 get_final_message() 取，计入 token 预算

### 横切面

- **trace**：request_id（13 位毫秒时间戳 + 4 位随机 hex）贯穿全链，span 落 SQLite。stage 口径：`pre_*/embed/search/retrieve/gen_first_token/gen_total/e2e/ingest`，status = ok/degraded/error——排障、压测聚合、README 数字同一口径。不上 OTel 全家桶，但字段留了升级路
- **limits**：50 问/人/天（`DAILY_QUESTION_LIMIT`）+ 500k token/天全局（`DAILY_TOKEN_BUDGET`），计数落 SQLite 按自然日重置；超限给友好提示（429 / 钉钉文本），不是报错——限流是产品行为不是故障
- **media**：图片原件 `data/media/`，chunk 只存文件名，前端经 `/api/media/` 回显

### 模型拓扑（key 拆分：生成商可独立切换，检索侧固定智谱）

| env | 指向 | 用途 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 智谱 | embedding-3（1024 维）/ rerank（已退役留 flag）/ glm-4v-flash（图片描述） |
| `CHAT_API_KEY` | DeepSeek anthropic 兼容端点 | 生成 + pre-retrieval 小生成，默认 `deepseek-flash`（`ANTHROPIC_BASE_URL` / `ANTHROPIC_CHAT_MODEL` 可换别家）——实测快 4 倍且评测持平 |

## 评测（数字真实跑出来，复现命令都在本仓）

```bash
python -X utf8 -m eval.run_eval                 # 22 条 × 每条生成 3 次
python -X utf8 -m eval.run_eval --pre rewrite   # pre-retrieval 四模式消融
```

### 评测集设计（eval/evalset.py：22 条，四层 × 模态 tag）

| 层 | 条数 | 考什么 |
|---|---|---|
| 常规 | 12 | 基本盘：语料里有、问法直白，RAG 应该拿分。含 2 条锚定 2025/云天化新语料、M05 docx 标准样式验收、M07 docx 反样式真实 case（真数字+加粗伪装标题+合并单元格，验章节兜底路径） |
| 边界 | 6 | 加压形态：短查询 / 跨块聚合（答案分散两块，Top-k 只命中一块也会缺数）/ 图片定性 / 措辞转换（问「董事长」、语料只有「法定代表人」）/ 答案埋在行政文本 / 多轮指代（M06 带 history：off 裸检索「那2025年呢」必挂，改写消解指代后应过） |
| 综合 | 2 | B 类跨文档：两个公司的表都进 context 才答得全（组间 AND） |
| 异常 | 2 | 不该做的题：语料里没有（含世界知识有的「氯化钾化学式」——考「只能根据参考资料回答」的约束力），期望拒答不编造 |

每条带模态 tag（表格/正文/图片），指标可按模态拆开看哪类 chunk 检索最弱。

**收题规则**（v1→v5 教训沉淀，写在 evalset.py 头部）：

- **只收有唯一标准答案的题**。原「真实回流」层 4 条线上回收题（分红追问/扣非缩写/eps/口语）收集于单文档时代，6 文档扩容后「公司」无唯一指向——拿旧标准答案判分 = 判无解之题，已撤层；等线上回流带明确对象的问题再入集
- 题型分类约束：A 明确标准 / B 多文档综合 / C 目标缺失歧义（不入集——澄清反问是 bot 产品功能，当前不存在）/ D 其它（X01/X02 语料外拒答题即 D）
- **评测集自己的标签也会错**：v1 曾把「董事长是谁」「分红方案」标成语料外异常题，逐 case 核对语料后都翻了——每条「语料外」假设都要拿语料核对过再定标签

### 两层判分（全代码断言，不靠肉眼）

每条 case 两个判分层 + 异常题一个负向断言：

- **expect（检索层标准出处）**：**组间 AND、组内 OR**——跨文档题要求两个公司的表都进 Top-k。候选带 source 锚（多文档库中 type+page 会跨文档假命中，判分只认主语料的块）；异常题 expect=None 不判检索——检索总会返回「最不差的」Top-k，精排分再高也证不了「语料里没有」
- **assert_groups（生成层）**：每组关键词任一命中、全部组过 = 答对。断言关键词给全格式与缩略两种（"15,134,119,500" / "151.34"），两种写法都吃得住
- **refute（异常题负向断言）**：拒答词命中 **且** 拒答句之后没有编造内容才算过——「资料中未提及，但据我所知是 KCl」照样判错。词表只收动作词 + 拒答句内命中豁免（「未提及…发展战略…等内容」是元语言解释不是编造；10-30 复盘：模型实际 15/15 拒答，旧判分报 1/3 全是误伤）

**评测环境自带鲁棒性设计**：独立库 `data/eval.db`（锁与生产按文件隔离，无需停服务）；自动导入 sample_reports 全部样本 + 虚构公司年报干扰文档（`eval/distractor.txt`，考检索/归因不被无关文档带偏）；生成探针与线上 Web/bot 走**同一段流式代码**（首字超时重试/关思考同口径）。参数：`--runs`（每条生成次数，默认 3）、`--top-k`（context 消融）、`--pre`（四模式）、`--rerank`。

### 指标口径（按环节拆开——成功率上不去时分得清是检索没捞到还是生成没答好）

- **检索层**：命中@3（历史口径）/ 命中@5（=context 口径）/ Top-1。判据是「该 case 的标准出处**全部**进 Top-k」（组间 AND），本质是二元化的证据 recall@k；经典 recall 的分母需要全语料相关性标注，不可得也不必要——RAG 的目标是 context 不缺证据
- **生成层**：答对率（关键词组断言）/ 拒绝率（refute 加固后）/ pass@N、pass^N（每题 N 次生成，至少一次对 / 次次对）
- **过程与成本**：检索中位/P95 ms、生成中位/P95 s、token 合计与均值

### 当前基线（22 条 × 3 次，pre=off）

| 指标 | 数值 | 说明 |
|---|---|---|
| 检索命中@5 | 80.0% | 标准出处进 Top-5；命中@3 75.0%；分层@5：常规 36/36 / 边界 12/18 / 综合 0/6；Top-1 70.0% |
| 答对率 | 80.0% | 生成层关键词组断言；context 带类型+来源标注，模型能按公司/年份正确归因 |
| 拒绝率（不该做的题） | 100% | 语料外问题（含世界知识有的）零幻觉漏答；拒答却带出编造内容 0 次 |
| pass@3 / pass^3 | 16/20 · 16/20 | 每题生成 3 次，全对才记 pass^3 |
| 生成延迟 | p50 1.5s / P95 2.0s | deepseek-flash、关思考 |

（22 条口径 = 21 条 + M07 docx 反样式真实 case；命中绝对数 48/57→48/60 不变——分母变化 + 库重建的 VLM 描述方差，非回归；M05/M07 两类 docx 管线题全绿 ✓Top1。综合层检索 0/6 = 组间 AND 在 Top-5 太严，但 M02 靠 2025 表对比列照样答对——检索锚定比答案所需更严的活例。口径沿革：19 条无 source 锚时代命中@5 64.7% / 答对 ≈63%，单文档时代命中@3 76.5% / 答对 ≈70%，见演进记录）

### pre-retrieval 消融（`--pre off/rewrite/hyde/multi` 同库对比；rewrite=LLM 改写消解多轮指代，hyde=假设文档只换 dense 向量，multi=3 变体+原问四路 RRF 合并）

| 模式 | 命中@5 | Top-1 | 答对 | 拒绝 | 检索中位 | 均 token | 结论 |
|---|---|---|---|---|---|---|---|
| **off（默认）** | **84.2%** | 63.2% | **84.2%** | **100%** | **425ms** | **329** | 全指标并列最优或独享 → 默认 |
| rewrite | 84.2% | **73.7%** | 84.2% | 83.3% ⚠ | 1776ms | 793 | 仅 Top-1 +10.5pp，代价=编造泄漏 1 次+延迟×4.2 → flag 备用 |
| hyde | 78.9% | 63.2% | 80.7% | 100% | 2050ms | 1266 | 假设文档稀释实体词，净负 |
| multi | 82.5% | 66.7% | 82.5% | 100% | 2015ms | 1464 | 变体 fan-out 不敌 source 锚，净负 |

数据修对 > 检索花活：表格块 text 加 source 锚（+26.3pp）吞掉了 pre-retrieval 的几乎全部增益——先修数据再上技巧的活例（rerank 退役后第二个负结果案例，数字都在）。

### 压测（`python -X utf8 -m loadtest.bench`，60 请求 0 失败）

| 并发 | p50 | p95 | 吞吐 |
|---|---|---|---|
| 1 | 3.7s | 3.9s | 0.29 req/s |
| 4 | 3.4s | 9.4s | 0.80 req/s |
| 8 | 3.9s | 5.7s | 1.73 req/s |

分阶段延迟（tracing 聚合，8 并发窗口）：embed 0.47s · Milvus search 0.03s（热库）· rerank 0.57s · **首字 1.33s** · e2e p50 3.68s / P95 6.86s。成本 ≈ 458 token/请求。瓶颈在上游 API 延迟，服务本体并行无压力。

## 可靠性设计（每个数字都可调）

| 机制 | 参数 | 行为 |
|---|---|---|
| embedding / VLM 有限重试 | 2 次指数退避 | 幂等读类调用才重试 |
| rerank 层 | 已退役（10-29） | 智谱 rerank 分数压缩在 0.99999X 饱和（接口/参数无误），评测实证开着检索净 -11.8pp、延迟 +0.5s——检索直接走融合序；`use_rerank=True` 开关留作换供应商后 A/B |
| 生成首字超时 | 15s（httpx read timeout） | 取消重试一次——长尾多为上游排队抖动，重试大概率落快通道 |
| 生成总时长 | 90s 兜底 | 超时上抛，Web 走 SSE error / 钉钉走 ai_fail |
| pre-retrieval 层 | 5s 超时、不重试 | LLM 改写/假设文档/多路变体失败即降级原问直检（trace 记 degraded）——降级就是容错，与 rerank 同哲学 |
| 限流 | 50 问/人/天（`DAILY_QUESTION_LIMIT`） | 超限友好提示（429 / 钉钉文本），不是报错 |
| token 预算 | 500k/天全局（`DAILY_TOKEN_BUDGET`） | 防单人/单日烧穿配额 |
| 追踪 | request_id 贯穿全链 | span 落 SQLite，按 stage 聚合 p50/p95/降级数——排障、压测、巡检同一口径 |

## 快速开始

```bash
pip install -r requirements.txt
set ANTHROPIC_API_KEY=你的智谱key        # 检索侧三件（embedding/rerank/VLM）
set CHAT_API_KEY=你的deepseekkey          # 生成固定 DeepSeek（实测快 4 倍、评测持平）
set ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
set ANTHROPIC_CHAT_MODEL=deepseek-flash

python -X utf8 -m app.main              # → http://127.0.0.1:7860
```

语料：`sample_reports/` 下 6 份精选 PDF——盐湖股份 2023/2024/2025 + 云天化 2023/2024/2025（各 10 页左右，镜像结构：重要提示/释义/简介/主要会计数据/非经常性损益/行业与图表页，`tools/extract_pages.py` 可从年报全本重抽）+ 2 份 docx 样本（`tools/make_docx_sample.py`：星澜科技=虚构公司标准样式，验基础管线；万华化学=**真实数字（摘自 2025 年报 p7）+ 反样式排版**——加粗伪装标题/合并单元格表头，验章节提取兜底路径）。全部导入即可跨年跨公司问答（约 210 块）。评测：`-m eval.run_eval` 用独立库 data/eval.db（自动导入全部样本 PDF/docx + 干扰文档），与线上服务互不干扰、无需停服务；压测 `-m loadtest.bench` 需服务在跑。

**支持的文档格式**：PDF（PyMuPDF 自研多模态管线）/ docx（python-docx，与 PDF 同构的三类 chunk：段落正文、表格→HTML、内嵌图片→VLM 描述）/ 纯文本。旧版 .doc（OLE2 二进制）不支持，上传时明确引导转存 .docx（评估过 LibreOffice headless 转换路线：引入 700MB 级外部依赖，语料现实里没有 .doc，不做）。

**钉钉机器人**（可选）：`DINGTALK_CLIENT_ID/SECRET` 两 env 设了即随服务启动（Stream 出站连接，免公网 IP/域名/证书）。平台侧：创建应用 → 机器人能力 → Stream 模式 → 开通 `Card.Instance.Write` + `Card.Streaming.Write` 两枚权限。未认证个人团队即可用。

## 部署（Windows 常驻）

```bash
copy deploy\env.bat.example deploy\env.bat    # 填 key（此文件已 gitignore）
deploy\start.bat                              # 前台跑，先验证一遍
```

常驻用 NSSM 装成 Windows 服务（开机自启 + 崩溃自动拉起）：

```
nssm install ragbot "D:\Anaconda3\python.exe" "-X utf8 -m app.main"
nssm set ragbot AppDirectory D:\workspace\ragbot
nssm set ragbot AppStdout D:\workspace\ragbot\ragbot_server.log
nssm set ragbot AppStderr D:\workspace\ragbot\ragbot_server.log
（env 在 nssm set ragbot Environment 里配，或系统级环境变量）
nssm start ragbot
```

## 演进记录（关键事实与结论）

- **表格 source 锚（10-30）**：独立成块的表格 text 只有指标+数字没公司名，BM25/dense 都吃不到。text 加 `source：` 前缀：命中 57.9→84.2%（**+26.3pp**）、Top-1 47.4→63.2%。结论：多模态 chunk 脱离正文独立成块时，来源必须进可检索文本
- **pre-retrieval 三模式 A/B（10-30）**：off 全指标最优 → 默认；rewrite 仅 Top-1 +10.5pp 但有编造泄漏 + 延迟×4.2；hyde/multi 净负。三模式留 flag 备用。source 锚吞掉 pre 几乎全部增益——先修数据再上技巧
- **统一 Chunk 模型 + 章节元数据（10-30）**：chunk_type 双名收口一处定义；section 入独立块元数据与脚注展示（「文档｜章节 p页」）。章节 text 锚 A/B 全指标持平=中性 → 前缀不加（`RAGBOT_SECTION_ANCHOR=1` 可复验）
- **异常题判分器去假阳性（10-30）**：X01 复盘=模型实际 15/15 拒答、fail 全是判分器误伤（元语言解释 + 引语料原文撞词）。修复=refute 只收动作词 + 拒答句内豁免：拒答回 100%，真编造照拦
- **docx 导入（10-30）**：python-docx 三类 chunk 与 PDF 同构；+M05 标准样式验收、+M06 多轮指代信号弹（off 裸检索 0/3、rewrite 3/3）、+M07 反样式真实 case 验章节兜底路径
- **评测加固（10-29）**：+refute 负向断言、生成探针改与线上同段流式代码、+虚构公司干扰文档、评测迁独立库 eval.db——重跑数字持平确认
- **语料 1→6 份 + 多文档评测适配（10-29）**：判分钉死 source、问句 pin 公司、断言加年份组（「不送红股」曾误配云天化派现=假阳性）；Top-3→5 命中 +11.7pp → 生产 TOP_K=5
- **来源标注归因（10-29）**：context 每块加来源 + system 要求按公司/年份归因——答对 34.9%→52.9%（此前模型把盐湖数字安到云天化头上）
- **评测集 v5（10-29）**：回流层撤空（无唯一答案不入集）；答案唯一性全量审计（pin 公司 / pin 年份 / 跨源锚定组 OR）
- **rerank 退役（10-29）**：智谱 rerank 分数压缩在 0.99999X 饱和无区分度，开着检索净 -11.8pp、延迟 +0.5s——直接走融合序，`use_rerank` 留作换供应商后 A/B

## License

MIT
