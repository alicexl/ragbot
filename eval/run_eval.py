"""评测：让「检索质量」从感觉变成数字（评测集见同目录 evalset.py）

一次评测 = 给系统一个输入，用一段判分逻辑检查输出，看成没成：
- 多次试验：每条 case 独立跑 --runs 次（默认 3），每次走完整链路
- 判分器全用代码：检索层判「标准出处进没进 Top-3」，生成层判关键词组 / 拒答
- 指标按环节拆开：检索（命中@3、Top-1）与生成（答对率、拒绝率、pass@N / pass^N）
  分开报——成功率上不去时，才分得清是检索没捞到还是生成没答好

启动（前提：智谱 Key 已设、库里已导入语料 PDF；先停掉服务——Milvus 单进程锁）:
    python -X utf8 -m eval.run_eval             # 每条生成 3 次
    python -X utf8 -m eval.run_eval --runs 5
"""

import argparse
import statistics
import time

from anthropic import Anthropic

from app import llm, pipeline
from app.store import TOP_K
from eval.evalset import CASES, REFUSE_PATTERNS


def _percentile(xs: list[float], p: int) -> float:
    """nearest-rank 百分位：P95 看尾部——单次慢不说明什么，5% 的请求慢才是问题"""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.01 * p * len(xs)))]


def _match(entry: dict, cand: dict) -> bool:
    """检索层命中判定：表格/图片候选按 类型+页码，正文候选按 类型+关键词锚定"""
    if entry["type"] != cand["type"]:
        return False
    if "kw" in cand:
        return cand["kw"] in entry["text"]
    return entry["page"] == cand["page"]


def _score_answer(answer: str, case: dict) -> bool:
    """生成层判分：普通题 = 每组关键词任一命中、全部组通过；异常题 = 判拒答"""
    if case["assert_groups"] is None:
        return any(p in answer for p in REFUSE_PATTERNS)
    return all(any(k in answer for k in group) for group in case["assert_groups"])


def generate(client: Anthropic, question: str, history: list[dict], retrieved: list[dict]):
    """非流式生成探针：拼装口径与线上 chat / bot 完全同构，messages.create 一把拿
    全文 + token 用量。glm-5.3 会带 ThinkingBlock，只取文本块。"""
    resp = client.messages.create(model=llm._CHAT_MODEL, max_tokens=2048,
                                  system=llm.SYSTEM_PROMPT,
                                  messages=pipeline.build_user_msg(question, retrieved, history))
    answer = "".join(b.text for b in resp.content if b.type == "text")
    return answer, resp.usage.input_tokens + resp.usage.output_tokens


def main():
    ap = argparse.ArgumentParser(description="ragbot 评测")
    ap.add_argument("--runs", type=int, default=3,
                    help="每条 case 的生成次数（pass@N / pass^N 的 N，默认 3）")
    args = ap.parse_args()

    pipeline.init()
    if not pipeline.store.chunk_count():
        raise SystemExit("向量库为空：先导入 sample_reports/盐湖股份2024年报（精选页）.pdf 再跑评测")
    client = Anthropic(base_url=llm._CHAT_BASE_URL, api_key=llm._CHAT_API_KEY)

    n_do = sum(1 for c in CASES if c["expect"] is not None)        # 该做的题（检索层也判）
    n_refuse = len(CASES) - n_do
    print("=" * 68)
    print(f"ragbot 评测 · 库内 {pipeline.store.chunk_count()} 块 · 评测集 {len(CASES)} 条"
          f"（该做 {n_do} / 不该做 {n_refuse}）· 每条生成 {args.runs} 次")
    print("=" * 68)

    rows = []
    ret_hits = ret_top1 = ret_runs = 0
    ret_by_layer: dict[str, list[int]] = {}
    gen_pass = gen_runs = 0
    refuse_pass = refuse_runs = 0
    degraded = 0
    ret_ms: list[float] = []
    gen_s: list[float] = []
    in_tok = out_tok = 0

    for case in CASES:
        hit_any = top1_any = False
        k_passed = 0
        for _ in range(args.runs):
            t0 = time.perf_counter()
            retrieved, reranked = pipeline.retrieve(case["question"], top_k=TOP_K)
            ret_ms.append((time.perf_counter() - t0) * 1000)
            degraded += (not reranked)

            if case["expect"] is not None:
                ret_runs += 1
                ret_by_layer.setdefault(case["layer"], [0, 0])[1] += 1
                hit = any(_match(e, c) for e in retrieved for c in case["expect"])
                top1 = any(_match(retrieved[0], c) for c in case["expect"])
                ret_hits += hit
                ret_top1 += top1
                if hit:
                    hit_any = True
                    ret_by_layer[case["layer"]][0] += 1
                top1_any = top1_any or top1

            t0 = time.perf_counter()
            answer, tokens = generate(client, case["question"], [], retrieved)
            gen_s.append(time.perf_counter() - t0)
            in_tok += tokens
            ok = _score_answer(answer, case)
            k_passed += ok
            if case["assert_groups"] is None:
                refuse_runs += 1
                refuse_pass += ok
            else:
                gen_runs += 1
                gen_pass += ok

        rows.append((case, hit_any, top1_any, k_passed))
        ret_mark = "—" if case["expect"] is None else \
            ("✓Top1" if top1_any else ("✓@3" if hit_any else "✗"))
        print(f"  {case['id']} [{case['layer']}·{case['tag']}] 检索 {ret_mark}"
              f" · 生成 {k_passed}/{args.runs} · {case['note'].split('，')[0].split('：')[0]}")

    pass_at = sum(1 for c, _, _, k in rows if c["assert_groups"] is not None and k >= 1)
    pass_all = sum(1 for c, _, _, k in rows if c["assert_groups"] is not None and k == args.runs)
    n_gen_cases = sum(1 for c in CASES if c["assert_groups"] is not None)

    print("=" * 68)
    print("检索层（标准出处进 Top-3 判命中；异常题无标准出处，不判）")
    print(f"  命中@3  {ret_hits}/{ret_runs}（{ret_hits / ret_runs:.1%}）"
          f" · Top-1  {ret_top1}/{ret_runs}（{ret_top1 / ret_runs:.1%}）")
    print("  分层（按 run 计）：" + " · ".join(
        f"{k} {v[0]}/{v[1]}" for k, v in ret_by_layer.items()))
    print("生成层（代码断言）")
    print(f"  该做的题   答对 {gen_pass}/{gen_runs}（{gen_pass / gen_runs:.1%}）")
    print(f"  不该做的题 拒绝 {refuse_pass}/{refuse_runs}（{refuse_pass / refuse_runs:.1%}）")
    print(f"  pass@{args.runs}  {pass_at}/{n_gen_cases} · "
          f"pass^{args.runs}  {pass_all}/{n_gen_cases}")
    print("过程与成本")
    print(f"  rerank 降级 {degraded}/{len(ret_ms)} · "
          f"检索 中位 {_percentile(ret_ms, 50):.0f}ms / P95 {_percentile(ret_ms, 95):.0f}ms · "
          f"生成 中位 {_percentile(gen_s, 50):.1f}s / P95 {_percentile(gen_s, 95):.1f}s")
    print(f"  token（合计 {in_tok}）· {len(gen_s)} 次生成")


if __name__ == "__main__":
    main()
