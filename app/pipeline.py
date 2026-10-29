"""RAG 管线胶合层：单例接线 + 检索/增强/生成入口

单例（embedder / splitter / store / reranker / generator）由 init() 在入口接好线，
web 路由、钉钉 bot、评测脚本三个调用方共享同一套——Milvus 单进程锁决定了
全服务必须单进程，模块级单例 + 调用时属性访问是最直白的形态。

检索链路：pre-retrieval（PRE_MODE，可选）→ embed → Milvus 两路召回 + RRF 融合 Top-20
→ rerank 精排 Top-3（失败降级融合序）。生成拼装口径（【参考资料】+【用户问题】）
全调用方同构——评测测的就是线上这条链。
"""

import os
import time
from dataclasses import dataclass

from . import llm, trace
from .store import (DATA_DIR, MEDIA_DIR, RECALL_K, TOP_K, Chunk, MilvusStore,
                    RecursiveTextSplitter, parse_docx, parse_pdf)


@dataclass
class RetrievedChunk(Chunk):
    """检索命中：统一 chunk 模型 + 检索层元数据（rrf=融合分展示用；score=rerank 分，
    rerank 退役后恒 None）。web / bot / eval 三个消费方共用。"""

    rrf: float = 0.0
    score: float | None = None

# Pre-retrieval 模式（互斥，一次只有一个生效）：off=原样 embed（基线）/ rewrite=LLM
# 改写问题（带 history 消解指代）/ hyde=假设文档只换 dense 向量（BM25 留原问——
# 假设文档分词偏，字面路用原问更稳）/ multi=3 变体+原问四路检索、RRF 合并。
# 默认 off（10-30 A/B 拍板，表格 source 锚修复后）：off 命中/答对 84.2% 并列最高、
# 拒绝 100%、检索 425ms/329tok 全场最低；rewrite 仅 Top-1 +10.5pp 但拒绝 83.3%
# （X01 编造泄漏 1 次）+ 延迟×4.2 + token×2.4；hyde 78.9%/multi 82.5% 净负。
# 三模式留 flag 备用（换更强改写模型 / 指代题变多时重启 A/B），rerank 同款待遇。
PRE_MODE = os.environ.get("RAGBOT_PRE_MODE", "off")

# 进程级单例：None 直到 init()（import 本模块零副作用）
embedder = None
splitter = None
store = None
reranker = None
generator = None


def init(db_path: str | None = None):
    """入口调用：配置检查 → 初始化全部单例。返回 None（失败直接抛 SystemExit）。
    db_path 不传 = 生产库 data/mini_rag.db；评测传 eval.db，与生产数据隔离。"""
    global embedder, splitter, store, reranker, generator
    if not llm._ZHIPU_API_KEY:
        raise SystemExit("未配置智谱 API Key（ANTHROPIC_API_KEY）——embedding/rerank/VLM 都依赖它")
    generator = llm.Generator()
    if not generator.api_key:
        raise SystemExit("未配置生成 API Key（CHAT_API_KEY）")
    embedder = llm.ZhipuEmbedder(llm._ZHIPU_API_KEY)
    splitter = RecursiveTextSplitter(chunk_size=400, chunk_overlap=50)
    reranker = llm.Reranker(llm._ZHIPU_API_KEY)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    store = MilvusStore(db_path or str(DATA_DIR / "mini_rag.db"))


def _to_entries(hits: list[dict], order: list[int], scores: dict[int, float]) -> list[RetrievedChunk]:
    """Milvus 命中 + 精排顺序 → RetrievedChunk（chat 路由与 build_context 共用）。"""
    return [RetrievedChunk(text=h["entity"]["text"], content=h["entity"]["content"],
                           chunk_type=h["entity"]["chunk_type"], page=h["entity"]["page"],
                           section=h["entity"].get("section", ""),
                           source=h["entity"]["source"],
                           rrf=h["distance"], score=scores.get(i))
            for i in order for h in [hits[i]]]


def retrieve(question: str, top_k: int = TOP_K, rid: str | None = None,
             use_rerank: bool = False, history: list[dict] | None = None,
             pre: str | None = None):
    """混合检索：两路召回 + RRF 融合 Top-20 → 直接取 Top-K。
    pre（默认取 PRE_MODE）：off/rewrite/hyde/multi 互斥——rewrite 两路 query 都换成
    改写问；hyde 只换 dense 向量（假设文档分词偏，BM25 留原问）；multi 四路检索后
    跨列表 RRF 合并（1/(60+rank) 求和，多路共识的块排上去）。LLM 失败/超时一律
    降级原问直检（trace 记 degraded）。
    rerank 层默认关闭——10-29 评测实证：智谱 rerank 分数压缩在 0.99999X 饱和
    （接口/参数无误，微差排序对噪声敏感），开着检索 -11.8pp、延迟 +0.5s；
    use_rerank=True 留作换 rerank 供应商后再 A/B。返回 (entries, reranked)。
    rid 不传则自动生成——每次检索都有 span 可查。"""
    rid = rid or trace.new_request_id()
    t_all = time.perf_counter()
    mode = pre or PRE_MODE
    if mode not in ("off", "rewrite", "hyde", "multi"):
        raise ValueError(f"未知 pre 模式：{mode}（off/rewrite/hyde/multi）")

    queries = [question]      # BM25 路用的查询文本（multi 时为变体列表）
    vec_texts = [question]    # dense 路要向量化的文本（hyde 时为假设文档）
    if mode != "off":
        t = time.perf_counter()
        try:
            if mode == "rewrite":
                queries = vec_texts = [llm.rewrite_query(question, history)]
            elif mode == "hyde":
                vec_texts = [llm.hypothetical_doc(question)]
            else:
                queries = vec_texts = llm.multi_query(question) + [question]
            trace.record(rid, f"pre_{mode}", (time.perf_counter() - t) * 1000)
        except Exception as e:
            queries = vec_texts = [question]
            trace.record(rid, f"pre_{mode}", (time.perf_counter() - t) * 1000,
                         status="degraded", meta={"error": str(e)[:80]})

    t = time.perf_counter()
    vecs = embedder.embed_batch(vec_texts)
    trace.record(rid, "embed", (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    if len(vecs) == 1:
        hits = store.hybrid_search(queries[0], vecs[0], top_k=RECALL_K)
    else:                      # multi：每变体一路（变体+原问），跨列表 RRF 合并去重
        scores: dict[int, float] = {}
        by_id: dict[int, dict] = {}
        for q, v in zip(queries, vecs):
            for rank, h in enumerate(store.hybrid_search(q, v, top_k=RECALL_K)):
                hid = h["id"]
                scores[hid] = scores.get(hid, 0.0) + 1.0 / (60 + rank)
                by_id.setdefault(hid, h)
        hits = [by_id[hid] for hid in sorted(scores, key=scores.get, reverse=True)]
    trace.record(rid, "search", (time.perf_counter() - t) * 1000)

    if not use_rerank:
        entries = _to_entries(hits, list(range(top_k)), {})
        trace.record(rid, "retrieve", (time.perf_counter() - t_all) * 1000)
        return entries, False

    docs = [h["entity"]["text"] for h in hits]
    t = time.perf_counter()
    try:
        results = reranker.rerank(question, docs, top_n=top_k)
        trace.record(rid, "rerank", (time.perf_counter() - t) * 1000)
        order = [r["index"] for r in results]
        entries = _to_entries(hits, order, {r["index"]: r["relevance_score"] for r in results})
        reranked = True
    except Exception as e:
        trace.record(rid, "rerank", (time.perf_counter() - t) * 1000,
                     status="degraded", meta={"error": str(e)[:80]})
        entries = _to_entries(hits, list(range(top_k)), {})
        reranked = False
    trace.record(rid, "retrieve", (time.perf_counter() - t_all) * 1000)
    return entries, reranked


def build_context(retrieved: list[RetrievedChunk]) -> str:
    """把精排后的 Top-K 拼成带编号 + 类型 + 来源标注的 context（回填 content 原文：
    表格=HTML、图片=描述）。来源标注让模型在多文档下按公司/年份正确归因——
    10-29 实测：无来源标注时模型把盐湖的数字安到云天化头上（M01/M03 探针）。"""
    parts = [f"[{i}][{llm._TYPE_LABEL.get(e.chunk_type, '正文')}｜来源：{e.source}] "
             f"{e.content or e.text}" for i, e in enumerate(retrieved, 1)]
    return "\n\n".join(parts)


def build_user_msg(question: str, retrieved: list[dict], history: list[dict] | None = None) -> list[dict]:
    """生成的 messages 拼装（全调用方同构）：多轮 history + 本轮带参考资料的问题。"""
    context = build_context(retrieved)
    user_msg = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"
    return (history or []) + [{"role": "user", "content": user_msg}]


def ingest_text(text: str, source: str) -> tuple[int, bool, str]:
    """切块 + 向量化 + 入库一份纯文本文档。幂等：source 已在库则跳过。"""
    if store.has_source(source):
        return 0, False, f"[SKIP]「{source}」已在库中"
    rid = trace.new_request_id()
    t0 = time.perf_counter()
    texts = splitter.split_text(text)
    chunks = [Chunk(text=c, chunk_type="text", content=c, page=0) for c in texts]
    store.insert_rows(chunks, embedder.embed_batch(texts), source)
    trace.record(rid, "ingest", (time.perf_counter() - t0) * 1000,
                 meta={"source": source, "chunks": len(texts)})
    return len(texts), True, f"[OK]「{source}」+{len(texts)} 块"


def _ingest_rows(source: str, chunks: list[Chunk], vlm_ok: int) -> tuple[int, bool, str]:
    """多模态 chunk 入库 + ingest span + 结果摘要（ingest_pdf / ingest_docx 共用尾巴）。"""
    rid = trace.new_request_id()
    t0 = time.perf_counter()
    store.insert_rows(chunks, embedder.embed_batch([c.text for c in chunks]), source)
    by_type = {t: sum(1 for c in chunks if c.chunk_type == t) for t in ("text", "table", "image")}
    imgs = by_type["image"]
    vlm_note = f"，VLM 描述 {vlm_ok}/{imgs}" if imgs else ""
    labels = {"text": "文本", "table": "表格", "image": "图片"}
    types = " + ".join(f"{v}{labels[k]}" for k, v in by_type.items() if v)
    trace.record(rid, "ingest", (time.perf_counter() - t0) * 1000,
                 meta={"source": source, "chunks": len(chunks), "vlm_ok": vlm_ok})
    return len(chunks), True, f"[OK]「{source}」+{len(chunks)} 块（{types}{vlm_note}）"


def ingest_pdf(data: bytes, source: str) -> tuple[int, bool, str]:
    """多模态分支（PDF）：解析 → 向量化 → 入库。幂等同上。"""
    if store.has_source(source):
        return 0, False, f"[SKIP]「{source}」已在库中"
    rows, vlm_ok = parse_pdf(data, source, llm._ZHIPU_API_KEY, splitter)
    if not rows:
        return 0, True, f"[WARN]「{source}」未解析出内容（扫描件？本服务只吃有文字层的 PDF）"
    return _ingest_rows(source, rows, vlm_ok)


def ingest_docx(data: bytes, source: str) -> tuple[int, bool, str]:
    """Word 分支（.docx）：与 PDF 管线同构（段落/表格/图片三类 chunk）。幂等同上。"""
    if store.has_source(source):
        return 0, False, f"[SKIP]「{source}」已在库中"
    rows, vlm_ok = parse_docx(data, source, llm._ZHIPU_API_KEY, splitter)
    if not rows:
        return 0, True, f"[WARN]「{source}」未解析出内容（空文档？）"
    return _ingest_rows(source, rows, vlm_ok)
