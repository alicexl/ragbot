"""压测：并发打 /api/chat，量延迟 / 吞吐 / 成本

用法（先确保服务在跑，且没有别的进程占 Milvus 锁）：
    python -X utf8 -m loadtest.bench                        # 并发 1/2/4/8，每档 4 问
    python -X utf8 -m loadtest.bench --levels 1,4 --per 6

输出：每档并发的 p50/p95/p99 / 错误数 / 吞吐；结束后按 trace.summary 聚合
**分阶段延迟**（embed/search/rerank/gen_first_token/e2e，与线上排障同一口径），
token 成本从 e2e span 的 meta 汇总——压测数据天然进了 tracing，不用重复埋点。
"""

import argparse
import json
import sqlite3
import statistics
import threading
import time

import requests

from app.trace import TRACE_DB

# 压测问题取评测集的常规 8 条（稳定、答案短、不触限流口径内的极端形态）
QUESTIONS = [
    "公司2024年营业收入和归母净利润分别是多少？",
    "2024年归母净利润比上一年下降了多少？",
    "公司去年赚了多少钱？",
    "2024年基本每股收益是多少？",
    "年报里的「公司」指的是哪家企业？",
    "我国钾肥的进口依存度大概是多少？",
    "全球钾肥产能主要集中在哪几家公司？",
    "全球锂矿储量的分布情况如何？前几名的占比是多少？",
]


def one_request(host: str, q: str, timeout: float) -> tuple[float, bool]:
    """一次问答：POST /api/chat，SSE 读到 done 帧算成功。返回 (毫秒, 成功否)"""
    t0 = time.perf_counter()
    cur, ok = None, False
    try:
        r = requests.post(f"{host}/api/chat", json={"message": q, "history": []},
                          timeout=timeout, stream=True)
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("event:"):
                cur = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and cur == "done":
                ok = True
                break
    except Exception:
        pass
    return (time.perf_counter() - t0) * 1000, ok


def run_level(host: str, level: int, per: int, timeout: float) -> tuple[list[float], int, float]:
    """一档并发：level 个线程各打 per 个问题（round-robin 取题）。返回 (延迟列表, 失败数, 墙钟)"""
    results: list[tuple[float, bool]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(level)

    def worker(wid: int):
        barrier.wait()                                  # 同时起跑，压出真实并发
        local = []
        for i in range(per):
            q = QUESTIONS[(wid * per + i) % len(QUESTIONS)]
            local.append(one_request(host, q, timeout))
        with lock:
            results.extend(local)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(level)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ms = [m for m, _ in results]
    fails = sum(1 for _, ok in results if not ok)
    return ms, fails, wall


def tokens_since(ts: float) -> int:
    """压测窗口内 e2e span 的 token 合计（成本口径）"""
    with sqlite3.connect(TRACE_DB) as c:
        rows = c.execute("SELECT meta FROM spans WHERE stage='e2e' AND ts>=?", (ts,)).fetchall()
    total = 0
    for (meta,) in rows:
        try:
            m = json.loads(meta) if meta else {}
            total += m.get("tokens", 0)
        except json.JSONDecodeError:
            pass
    return total


def main():
    ap = argparse.ArgumentParser(description="ragbot 压测")
    ap.add_argument("--host", default="http://127.0.0.1:7860")
    ap.add_argument("--levels", default="1,2,4,8", help="并发档位（逗号分隔）")
    ap.add_argument("--per", type=int, default=4, help="每线程请求数")
    ap.add_argument("--timeout", type=float, default=180, help="单请求超时（秒）")
    args = ap.parse_args()

    t_start = time.time()
    print(f"压测目标 {args.host} · 问题池 {len(QUESTIONS)} 条 · 每档并发 × {args.per} 请求")
    print("=" * 72)
    all_ms = []
    for level in [int(x) for x in args.levels.split(",")]:
        ms, fails, wall = run_level(args.host, level, args.per, args.timeout)
        all_ms.extend(ms)
        ms_sorted = sorted(ms)
        n = len(ms_sorted)
        print(f"并发 {level:>2} · {n} 请求 · 失败 {fails}"
              f" · p50 {ms_sorted[n // 2] / 1000:5.1f}s"
              f" · p95 {ms_sorted[min(n - 1, int(0.95 * n))] / 1000:5.1f}s"
              f" · p99 {ms_sorted[min(n - 1, int(0.99 * n))] / 1000:5.1f}s"
              f" · 吞吐 {n / wall:4.2f} req/s")
    print("=" * 72)
    print(f"全程 {len(all_ms)} 请求 · token 合计 {tokens_since(t_start):,}"
          f"（均 {tokens_since(t_start) // max(len(all_ms), 1):,}/请求）")
    print("分阶段延迟（trace 聚合，压测窗口）：")
    from app import trace
    for row in trace.summary(since_ts=t_start):
        if row["stage"] == "ingest":
            continue
        print(f"  {row['stage']:16s} n={row['n']:<4} p50={row['p50'] / 1000:6.2f}s"
              f" p95={row['p95'] / 1000:6.2f}s max={row['max'] / 1000:6.2f}s"
              f" degraded={row['degraded']} error={row['error']}")


if __name__ == "__main__":
    main()
