# ragbot — 能上线的多模态 RAG 问答服务

文档入库（PDF 正文 / 表格 / 图片三模态）→ 混合检索（稠密向量 + BM25 两路召回、RRF 融合、Cross-Encoder 精排）→ 流式生成。**Web 聊天（SSE 打字机）+ 钉钉机器人（Stream 模式免公网）双客户端**，带完整的生产化横切面：评测集、请求级追踪、超时/有限重试/降级、限流与 token 预算、压测数据。

> 源于教学项目 [Mini-RAG](https://github.com/alicexl/Mini-RAG)（四步从零搭 RAG 的原理线）；本仓是其中产品线的独立演化——检索/解析内核经过 Mini-RAG 全链路验证后迁来，围绕「能上线」补齐工程横切面。

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

**模型拓扑**（key 拆分，生成商可独立切换）：智谱 embedding-3 / rerank / glm-4v（检索侧固定）+ 生成走 anthropic 兼容端点（默认智谱 glm-5.3，env 三件套可切 DeepSeek——本项目实测 deepseek-flash 快 4 倍且评测持平，见下）。

## 关键指标（真实跑出来的，复现命令都在本仓）

**质量评测**（`python -X utf8 -m eval.run_eval`，19 条分层评测集 × 3 次，判分全代码断言）：

| 指标 | 数值 | 说明 |
|---|---|---|
| 检索命中@3 | 76.5% | 标准出处进 Top-3；分层：常规 87.5% / 边界 60% / 真实 75% |
| 答对率 | 76.5% | 生成层关键词组断言 |
| 拒绝率（不该做的题） | 100% | 语料外问题（含世界知识有的）零幻觉漏答 |
| 生成延迟 | p50 1.4s / P95 2.7s | deepseek-flash、关思考 |

评测集含「线上真实问题回流」层——机器人实际收到的问题（中英缩写「eps」、脱离上下文的追问、极简口语）原样入集做回归，包括一条**真实发生过的假拒答**（答案在库里但检索捞不到 → 模型如实说没有——召回假阴性的活案例）。

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
# 可选：生成切 DeepSeek（实测快 4 倍、评测持平）
set CHAT_API_KEY=你的deepseekkey
set ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
set ANTHROPIC_CHAT_MODEL=deepseek-flash

python -X utf8 -m app.main              # → http://127.0.0.1:7860
```

导入 `sample_reports/盐湖股份2024年报（精选页）.pdf`（35 块：20 正文 + 4 表格 + 11 图片）即可问答。评测与压测：先停服务（Milvus Lite 单进程锁），再跑 `-m eval.run_eval` / `-m loadtest.bench`（压测需服务在跑）。

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

- 检索嵌入从本地 Ollama（GTX970，单查 2.2s）换智谱 embedding-3：检索 2.7s → 0.9s，命中@3 持平 76.9%
- 生成选型三轮：glm-5.3 → glm-5.3-flash（评测否决：更慢且质量掉）→ deepseek-flash（生成 5.9s→1.4s、P95 19.8s→2.7s、token 省 63%，质量持平）——「用评测而不是感觉做技术决策」的完整样本
- 钉钉打字机三要素（首帧置 INPUTING / 中间帧直发 / isFinalize 终帧），缺一不可——SDK 的 `ai_streaming` 永不收尾，卡片会挂在流式会话只渲染第一帧

## License

MIT
