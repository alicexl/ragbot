"""轻量 tracing：request_id 贯穿全链路 + span 落 SQLite + 结构化日志

设计取舍：不上 OpenTelemetry/Jaeger 全家桶（单机服务用不着重型件），但 span 的
命名与字段留了升级路（request_id / stage / status / ms / meta，接 OTel 就是换个 exporter）。

stage 约定（压测与 README 按同一口径聚合）：
  pre_rewrite/…    pre-retrieval（PRE_MODE：pre_rewrite/pre_hyde/pre_multi；
                   status=degraded 表示 LLM 失败降级原问直检）
  embed            查询向量化
  search           Milvus 两路召回 + RRF 融合
  rerank           精排（status=degraded 表示失败降级回融合序）
  retrieve         检索整段（embed+search+rerank）
  gen_first_token  生成首字（打字机体验的关键指标）
  gen_total        生成整段
  e2e              一次请求端到端
  ingest           文档入库（meta 记 source 与块数）
"""

import logging
import random
import sqlite3
import time
from pathlib import Path

# 不从 store 导入 DATA_DIR（store → llm 会与 llm → trace 成环）：data 目录本模块自算
TRACE_DB = str(Path(__file__).parent.parent / "data" / "traces.db")

log = logging.getLogger("ragbot.trace")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [TRACE] %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


def new_request_id() -> str:
    """13 位毫秒时间戳 + 4 位随机 hex：可读、按时间可排、基本不撞"""
    return f"{int(time.time() * 1000):013d}{random.randint(0, 0xffff):04x}"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(TRACE_DB)
    c.execute("""CREATE TABLE IF NOT EXISTS spans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        ms REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'ok',      -- ok | degraded | error
        meta TEXT,                              -- 自由 JSON（重试次数/来源/条数…）
        ts REAL NOT NULL)""")
    return c


def record(request_id: str, stage: str, ms: float, status: str = "ok", meta: dict | None = None):
    """记一个 span：SQLite（可聚合查询）+ 结构化日志（实时看现场）"""
    import json as _json
    with _conn() as c:
        c.execute("INSERT INTO spans(request_id, stage, ms, status, meta, ts) VALUES(?,?,?,?,?,?)",
                  (request_id, stage, ms, status,
                   _json.dumps(meta, ensure_ascii=False) if meta else None, time.time()))
    extra = f" meta={meta}" if meta else ""
    log.info("%s %s %.0fms %s%s", request_id, stage, ms, status, extra)


def spans(request_id: str) -> list[tuple]:
    """按请求查整条 span 链（排障用：一次请求每段花了多久、哪段降级）"""
    with _conn() as c:
        return c.execute("SELECT stage, ms, status, meta FROM spans "
                         "WHERE request_id=? ORDER BY id", (request_id,)).fetchall()


def summary(since_ts: float | None = None) -> list[dict]:
    """按 stage 聚合：次数 / 中位 / P95 / 降级与错误数（压测报告与日常巡检共用）"""
    q = ("SELECT stage, ms, status FROM spans" +
         (" WHERE ts>=?" if since_ts else ""))
    with _conn() as c:
        rows = c.execute(q, (since_ts,) if since_ts else ()).fetchall()
    by: dict[str, list] = {}
    for stage, ms, status in rows:
        by.setdefault(stage, []).append((ms, status))
    out = []
    for stage, items in sorted(by.items()):
        xs = sorted(ms for ms, _ in items)
        n = len(xs)
        out.append({"stage": stage, "n": n,
                    "p50": xs[n // 2], "p95": xs[min(n - 1, int(0.95 * n))],
                    "max": xs[-1],
                    "degraded": sum(1 for _, s in items if s == "degraded"),
                    "error": sum(1 for _, s in items if s == "error")})
    return out
