"""评测：让「检索质量」从感觉变成数字（评测集见同目录 evalset.py）

一次评测 = 给系统一个输入，用一段判分逻辑检查输出，看成没成：
- 多次试验：每条 case 独立跑 --runs 次（默认 3），每次走完整链路
- 判分器全用代码：检索层判「标准出处进没进 Top-3」，生成层判关键词组 / 拒答
  （异常题 = 拒答词命中 且 没有带出 refute 编造关键词——「未提及，但 KCl」照样判错）
- 指标按环节拆开：检索（命中@3、Top-1）与生成（答对率、拒绝率、pass@N / pass^N）
  分开报——成功率上不去时，才分得清是检索没捞到还是生成没答好

语料：评测用独立库 data/eval.db，自动导入 sample_reports/ 下全部 PDF/docx + 干扰文档
（eval/distractor.txt，虚构公司年报——考检索/归因不被无关文档带偏）。独立库不碰
生产库 mini_rag.db，锁按文件隔离，线上服务不用停；eval.db 可整库删除重建。

启动（前提：智谱 Key 已设）:
    python -X utf8 -m eval.run_eval             # 每条生成 3 次
    python -X utf8 -m eval.run_eval --runs 5
    python -X utf8 -m eval.run_eval --pre rewrite   # pre-retrieval A/B（off/rewrite/hyde/multi）
"""

import argparse
import os
import time
from pathlib import Path

from app import llm, pipeline, trace
from app.store import DATA_DIR, TOP_K
from eval.evalset import CASES, EVAL_SOURCE, REFUSE_PATTERNS

EVAL_DB = str(DATA_DIR / "eval.db")                           # 评测独立库，与生产隔离
SAMPLE_DIR = Path(__file__).parent.parent / "sample_reports"  # 样本 PDF（多放即多干扰文档）
DISTRACTOR = Path(__file__).parent / "distractor.txt"         # 虚构公司年报干扰语料


def _percentile(xs: list[float], p: int) -> float:
    """nearest-rank 百分位：P95 看尾部——单次慢不说明什么，5% 的请求慢才是问题"""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.01 * p * len(xs)))]


def _match(entry: dict, cand: dict) -> bool:
    """检索层命中判定：默认只认评测主语料（EVAL_SOURCE）的块——多文档库中表格/
    图片按 类型+页码、正文按 类型+关键词锚定，但都必须出自同一来源，否则
    2023/2025/云天化的同页码表格会假命中。候选带 source 时以其为准（锚定新语料）。"""
    if entry["source"] != cand.get("source", EVAL_SOURCE):
        return False
    if entry["type"] != cand["type"]:
        return False
    if "kw" in cand:
        return cand["kw"] in entry["text"]
    return entry["page"] == cand["page"]


def _expect_groups(case: dict) -> list[list[dict]]:
    """expect 归一化为组列表：候选列表（旧，单组 OR）或 组列表（新，组间 AND
    组内 OR——B 类综合题要求多个出处都进 context）。"""
    return case["expect"] if case["expect"] and isinstance(case["expect"][0], list) \
        else [case["expect"]]


def _score_answer(answer: str, case: dict) -> bool:
    """生成层判分：普通题 = 每组关键词任一命中、全部组通过；
    异常题 = 拒答 且 拒答句之后没有编造内容。refute 命中只看拒答句之后——
    拒答句内的命中是元语言解释（「未提及…发展战略…等内容」），不是编造。
    （10-30 X01 复盘：模型实际 15/15 拒答，eval fail 全是旧判分把元语言当编造的假阳性）"""
    if case["assert_groups"] is None:
        refuse_at = [answer.find(p) for p in REFUSE_PATTERNS if p in answer]
        if not refuse_at:
            return False
        first = min(refuse_at)
        sent_end = len(answer)
        for stop in "。！？":
            pos = answer.find(stop, first)
            if pos != -1:
                sent_end = min(sent_end, pos + 1)   # 拒答句边界（含句末标点）
        leak = any(r.lower() in answer[sent_end:].lower()
                   for r in case.get("refute", []))
        return not leak
    return all(any(k in answer for k in group) for group in case["assert_groups"])


def generate(question: str, history: list[dict], retrieved: list[dict]) -> tuple[str, int]:
    """流式生成探针：与线上 web / bot 同一段代码（llm.StreamedAnswer——首字超时重试 +
    总时长兜底、关思考），拼装口径同 pipeline.build_user_msg。评测测的就是线上这条链。"""
    sa = llm.StreamedAnswer(llm.SYSTEM_PROMPT,
                            pipeline.build_user_msg(question, retrieved, history),
                            rid=trace.new_request_id(), thinking_off=True)
    answer = "".join(sa)
    return answer, sa.usage


def _ensure_fixtures() -> str:
    """评测库语料：sample_reports/ 下全部 PDF/docx + distractor.txt（幂等，已入库自动跳过）。
    首次导入 PDF 会跑 VLM 描述图片，慢属正常；想加真干扰文档就往 sample_reports 丢 PDF。"""
    notes = []
    for name in sorted(os.listdir(SAMPLE_DIR)):          # 中文/特殊字符文件名，不用 glob
        if name.lower().endswith(".pdf"):
            notes.append(pipeline.ingest_pdf((SAMPLE_DIR / name).read_bytes(),
                                             Path(name).stem)[2])
        elif name.lower().endswith(".docx"):
            notes.append(pipeline.ingest_docx((SAMPLE_DIR / name).read_bytes(),
                                              Path(name).stem)[2])
    notes.append(pipeline.ingest_text(DISTRACTOR.read_text(encoding="utf-8"),
                                      DISTRACTOR.stem)[2])
    return " · ".join(notes)


def main():
    ap = argparse.ArgumentParser(description="ragbot 评测")
    ap.add_argument("--runs", type=int, default=3,
                    help="每条 case 的生成次数（pass@N / pass^N 的 N，默认 3）")
    ap.add_argument("--top-k", type=int, default=TOP_K,
                    help="检索进生成 context 的条数（默认=生产常量 TOP_K；仅评测实验，不改生产）")
    ap.add_argument("--rerank", action="store_true",
                    help="打开 rerank 层（默认关闭——10-29 评测实证净负，留作换供应商后 A/B）")
    ap.add_argument("--pre", choices=["off", "rewrite", "hyde", "multi"], default="off",
                    help="pre-retrieval 模式（默认 off=基线；只影响本次评测，不改生产 PRE_MODE）")
    args = ap.parse_args()

    pipeline.init(db_path=EVAL_DB)
    print("[fixtures]", _ensure_fixtures())
    if not pipeline.store.chunk_count():
        raise SystemExit("评测库为空：检查 sample_reports/ 与 eval/distractor.txt")

    n_do = sum(1 for c in CASES if c["expect"] is not None)        # 该做的题（检索层也判）
    n_refuse = len(CASES) - n_do
    print("=" * 68)
    print(f"ragbot 评测 · 独立库 {pipeline.store.chunk_count()} 块（含干扰文档）"
          f" · 评测集 {len(CASES)} 条（该做 {n_do} / 不该做 {n_refuse}）· 每条生成 {args.runs} 次"
          f" · context Top-{args.top_k} · pre={args.pre}")
    print("=" * 68)

    rows = []
    ret_hits = ret_hits_n = ret_top1 = ret_runs = 0
    ret_by_layer: dict[str, list[int]] = {}
    gen_pass = gen_runs = 0
    refuse_pass = refuse_runs = 0
    refuse_leak = 0
    degraded = 0
    ret_ms: list[float] = []
    gen_s: list[float] = []
    tok = 0

    for case in CASES:
        hit_any = top1_any = False
        k_passed = 0
        for _ in range(args.runs):
            t0 = time.perf_counter()
            retrieved, reranked = pipeline.retrieve(case["question"], top_k=args.top_k,
                                                    use_rerank=args.rerank,
                                                    history=case.get("history"),
                                                    pre=args.pre)
            ret_ms.append((time.perf_counter() - t0) * 1000)
            degraded += (not reranked)

            if case["expect"] is not None:
                ret_runs += 1
                ret_by_layer.setdefault(case["layer"], [0, 0])[1] += 1
                groups = _expect_groups(case)
                hit = all(any(_match(e, c) for e in retrieved[:3] for c in g) for g in groups)   # 历史口径 @3
                hit_n = all(any(_match(e, c) for e in retrieved for c in g) for g in groups)     # 上下文口径 @N
                top1 = any(_match(retrieved[0], c) for g in groups for c in g)
                ret_hits += hit
                ret_hits_n += hit_n
                ret_top1 += top1
                if hit_n:
                    hit_any = True
                    ret_by_layer[case["layer"]][0] += 1
                top1_any = top1_any or top1

            t0 = time.perf_counter()
            answer, tokens = generate(case["question"], case.get("history") or [], retrieved)
            gen_s.append(time.perf_counter() - t0)
            tok += tokens
            ok = _score_answer(answer, case)
            k_passed += ok
            if case["assert_groups"] is None:
                refuse_runs += 1
                refuse_pass += ok
                if not ok and any(p in answer for p in REFUSE_PATTERNS):
                    refuse_leak += 1      # 拒答词命中却判错 = 拒答后还带出编造内容（refute 命中）
            else:
                gen_runs += 1
                gen_pass += ok

        rows.append((case, hit_any, top1_any, k_passed))
        ret_mark = "—" if case["expect"] is None else \
            ("✓Top1" if top1_any else (f"✓@{args.top_k}" if hit_any else "✗"))
        print(f"  {case['id']} [{case['layer']}·{case['tag']}] 检索 {ret_mark}"
              f" · 生成 {k_passed}/{args.runs} · {case['note'].split('，')[0].split('：')[0]}")

    pass_at = sum(1 for c, _, _, k in rows if c["assert_groups"] is not None and k >= 1)
    pass_all = sum(1 for c, _, _, k in rows if c["assert_groups"] is not None and k == args.runs)
    n_gen_cases = sum(1 for c in CASES if c["assert_groups"] is not None)

    print("=" * 68)
    print("检索层（标准出处进 Top-3 判命中；异常题无标准出处，不判）")
    print(f"  命中@3  {ret_hits}/{ret_runs}（{ret_hits / ret_runs:.1%}）"
          f" · 命中@{args.top_k}  {ret_hits_n}/{ret_runs}（{ret_hits_n / ret_runs:.1%}）"
          f" · Top-1  {ret_top1}/{ret_runs}（{ret_top1 / ret_runs:.1%}）")
    print("  分层（按 run 计）：" + " · ".join(
        f"{k} {v[0]}/{v[1]}" for k, v in ret_by_layer.items()))
    print("生成层（代码断言）")
    print(f"  该做的题   答对 {gen_pass}/{gen_runs}（{gen_pass / gen_runs:.1%}）")
    print(f"  不该做的题 拒绝 {refuse_pass}/{refuse_runs}（{refuse_pass / refuse_runs:.1%}）"
          f" · 拒答却带出编造内容 {refuse_leak} 次")
    print(f"  pass@{args.runs}  {pass_at}/{n_gen_cases} · "
          f"pass^{args.runs}  {pass_all}/{n_gen_cases}")
    print("过程与成本")
    rerank_note = f"rerank 降级 {degraded}/{len(ret_ms)}" if args.rerank else "rerank 关闭（融合序）"
    print(f"  {rerank_note} · "
          f"检索 中位 {_percentile(ret_ms, 50):.0f}ms / P95 {_percentile(ret_ms, 95):.0f}ms · "
          f"生成 中位 {_percentile(gen_s, 50):.1f}s / P95 {_percentile(gen_s, 95):.1f}s")
    print(f"  token 合计 {tok:,} · 均 {tok // len(gen_s):,}/次 · {len(gen_s)} 次生成")


if __name__ == "__main__":
    main()
