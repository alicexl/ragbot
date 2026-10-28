# ragbot — 能上线的多模态 RAG 问答服务

文档入库（PDF 正文 / 表格 / 图片三模态）→ 混合检索（稠密向量 + BM25 两路召回、RRF 融合、Cross-Encoder 精排）→ 流式生成。**Web 聊天（SSE 打字机）+ 钉钉机器人（Stream 模式免公网）双客户端**，带完整的生产化横切面：评测集、请求级追踪、超时/有限重试/降级、限流与 token 预算、压测数据。

> 源于教学项目 [Mini-RAG](https://github.com/alicexl/Mini-RAG)（三步从零搭 RAG 的原理线）；本仓是其原 step4 应用的产品线独立演化——检索/解析内核在 Mini-RAG 内完成 E2E 与评测验证后迁出，围绕「能上线」补齐工程横切面。

## 架构

```
                    ┌─ Web 客户端（app/static，SSE 流式 + 文档上传）
请求 → Flask 路由 ──┤
                    └─ 钉钉机器人（app/bot.py，Stream 出站 WebSocket 免公网，
                       AI 卡片打字机，会话与使用日志服务端持有）
        │
   pipeline（app/pipeline.py：三客户端共享同一条链）
        ├─ ingest   PDF 多模态解析（表格→HTML 保行列、图片→VLM 描述）→ 切块 → embedding → Milvus
        ├─ retrieve embedding → Milvus 两路召回+RRF → rerank 精排（失败降级融合序）
        ├─ generate 拼参考资料 → 流式生成（首字超时重试一次 / 总时长兜底）
        │
   横切面：trace（request_id + span 落 SQLite）· limits（按人日限 + token 日预算）
```

**模型拓扑**（key 拆分）：智谱 embedding-3 / rerank / glm-4v（检索侧固定）+ 生成固定 DeepSeek anthropic 兼容端点（deepseek-flash，实测快 4 倍且评测持平，见下）。

## 关键指标（真实跑出来的，复现命令都在本仓）

**质量评测**（`python -X utf8 -m eval.run_eval`，19 条四层评测集 × 3 次，判分全代码断言；评测用独立库 data/eval.db + 干扰文档，无需停服务）：

| 指标 | 数值 | 说明 |
|---|---|---|
| 检索命中@5 | 52.9% | 标准出处进 Top-5（context 口径）；命中@3 41.2%；分层@5：常规 60% / 边界 60% / 综合 0%；Top-1 17.6% |
| 答对率 | 52.9%（两轮一致） | 生成层关键词组断言；context 带来源标注后模型能按公司/年份正确归因 |
| 拒绝率（不该做的题） | 100% | 语料外问题（含世界知识有的）零幻觉漏答；拒答却带出编造内容 0 次 |
| pass@3 / pass^3 | 9/17 · 9/17 | 每题生成 3 次，全对才记 pass^3 |
| 生成延迟 | p50 1.5s / P95 2.0s | deepseek-flash、关思考 |

以上为多文档口径（6 份精选 ≈205 块，19 条四层：常规10/边界5/综合2/异常2）；单文档时代（仅盐湖2024）命中@3 76.5% / 答对 ≈70%，见演进记录。综合层检索 0/6=组间 AND 在 Top-5 太严，但 M02 靠 2025 表对比列照样答对——检索锚定比答案所需更严的活例。

评测集只收有唯一标准答案的题：原「真实回流」层 4 条线上回收题（分红追问/扣非缩写/eps/极简口语）收集于单文档时代，6 文档扩容后无唯一答案——继续拿旧标准答案判分等于判无解之题，已撤层；等线上回流带明确对象（公司名/完整上下文）的问题再入集。题型分类约束（eval/evalset.py 头部）：构造题与回流题一视同仁，优先归 A 明确标准题 / B 多文档综合题 / C 目标缺失歧义题，不能归类的进 D 其它（X01/X02 即 D）。

评测集判分加固：异常题要求「拒答词命中 且 没有带出编造内容」（refute 负向断言——「资料中未提及，但 KCl」照样判错）；库内另有虚构公司年报（`eval/distractor.txt`）作干扰文档。评测生成与线上同一段流式代码（首字超时重试/关思考同口径）。多文档适配：检索判分钉死主语料 source（type+page 跨文档会假命中）、构造题带「盐湖股份」限定、断言加年份组；`--top-k` 参数可做 context 消融实验。

**压测**（`python -X utf8 -m loadtest.bench`，60 请求 0 失败）：

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
| rerank 快速失败 | 8s 超时，不重试 | **失败降级回 RRF 融合序**（span 记 degraded，精度损失可控） |
| 生成首字超时 | 15s（httpx read timeout） | 取消重试一次——长尾多为上游排队抖动，重试大概率落快通道 |
| 生成总时长 | 90s 兜底 | 超时上抛，Web 走 SSE error / 钉钉走 ai_fail |
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

语料：`sample_reports/` 下 6 份精选 PDF——盐湖股份 2023/2024/2025 + 云天化 2023/2024/2025（各 10 页左右，镜像结构：重要提示/释义/简介/主要会计数据/非经常性损益/行业与图表页，`tools/extract_pages.py` 可从年报全本重抽）。全部导入即可跨年跨公司问答（约 200 块）。评测：`-m eval.run_eval` 用独立库 data/eval.db（自动导入全部样本 PDF + 干扰文档），与线上服务互不干扰、无需停服务；压测 `-m loadtest.bench` 需服务在跑。

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

## 演进记录（都有评测数字背书）

- 生成选型三轮：glm-5.3 → glm-5.3-flash（评测否决：更慢且质量掉）→ deepseek-flash（生成 5.9s→1.4s、P95 19.8s→2.7s、token 省 63%，质量持平）——「用评测而不是感觉做技术决策」的完整样本；10-29 定案只保留 DeepSeek，代码默认与文档同口径
- 评测加固（10-29）：异常题加 refute 负向断言（「未提及，但 KCl」照样判错）、生成探针改走与线上同段流式代码、库内加虚构公司干扰文档、评测迁独立库 data/eval.db（无需停服务、不碰生产库）——重测检索 76.5% 持平、答对 ≈70%（旧口径 76.5% 系单轮幸运值，两轮复跑确认波动在边界题）、拒绝 100% 零编造
- 语料扩容与多文档评测适配（10-29）：sample_reports 1→6 份精选（盐湖股份 2023/2024/2025 + 云天化 2023/2024/2025，`tools/extract_pages.py` 抽页、结构镜像）；评测适配=检索判分钉死主语料 source（type+page 跨文档假命中）、构造题加公司名、断言加年份组（「不送红股」曾误配云天化派现方案=假阳性教训）、`--top-k` 消融参数；Top-3→5 两轮对比（命中 47.1→58.8%、答对 35.3→≈39%）→ 生产 TOP_K=5。剩硬骨头：B05 分红题（云天化派现块的检索信号强于盐湖行政文本）、行业题 C06/C08/B02
- B 类综合题 + 来源标注归因修复（10-29）：评测集 v3→v4（19→23 条，新增综合 2 条=B 类跨公司比较——expect 组形态组间 AND/组内 OR；常规新增 2 条锚定 2025/云天化，source 覆盖机制）；探针抓到归因失败（模型把盐湖数字安到云天化头上）→ context 每块加「来源」标注 + system prompt 要求按公司/年份归因——答对 34.9%→≈48%（两轮 47.6/49.2），检索不变（42.9%@3 / 52.4%@5）、拒绝 100%
- 评测集 v5（10-29）：真实层撤空——4 条回流题多文档后无唯一答案（判分=判无解之题），19 条四层定稿、两轮数字完全一致（命中@5 52.9% / 答对 52.9% / 拒绝 100%）；题型分类约束固化：构造与回流一视同仁归 A/B/C，不能归类进 D（X01/X02 即 D）；同日答案唯一性全量审计：C05 问句 pin「盐湖股份」（pin 后检索面从释义表变股票简况表，锚定改 kw 组 OR）、C06/C08 pin 年份（2023 版有竞争数字），B01 语料内唯一、C07 三版同事实，不动
- 钉钉打字机三要素（首帧置 INPUTING / 中间帧直发 / isFinalize 终帧），缺一不可——SDK 的 `ai_streaming` 永不收尾，卡片会挂在流式会话只渲染第一帧

## License

MIT
