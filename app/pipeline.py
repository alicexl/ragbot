"""RAG 管线胶合层：单例接线 + 检索/增强/生成入口

单例（embedder / splitter / store / reranker / generator）由 init() 在入口接好线，
web 路由、钉钉 bot、评测脚本三个调用方共享同一套——Milvus 单进程锁决定了
全服务必须单进程，模块级单例 + 调用时属性访问是最直白的形态。

检索链路：embed → Milvus 两路召回 + RRF 融合 Top-20 → rerank 精排 Top-3（失败降级融合序）。
生成拼装口径（【参考资料】+【用户问题】）全调用方同构——评测测的就是线上这条链。
"""

from . import llm
from .store import (DATA_DIR, MEDIA_DIR, RECALL_K, TOP_K, MilvusStore,
                    RecursiveTextSplitter, parse_pdf)

# 进程级单例：None 直到 init()（import 本模块零副作用）
embedder = None
splitter = None
store = None
reranker = None
generator = None


def init():
    """入口调用：配置检查 → 初始化全部单例。返回 None（失败直接抛 SystemExit）。"""
    global embedder, splitter, store, reranker, generator
    if not llm._ZHIPU_API_KEY:
        raise SystemExit("未配置智谱 API Key（ANTHROPIC_API_KEY）——embedding/rerank/VLM 都依赖它")
    generator = llm.Generator()
    if not generator.api_key:
        raise SystemExit("未配置生成 API Key（CHAT_API_KEY 或 ANTHROPIC_API_KEY）")
    embedder = llm.ZhipuEmbedder(llm._ZHIPU_API_KEY)
    splitter = RecursiveTextSplitter(chunk_size=400, chunk_overlap=50)
    reranker = llm.Reranker(llm._ZHIPU_API_KEY)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    store = MilvusStore(str(DATA_DIR / "mini_rag.db"))


def _to_entries(hits: list[dict], order: list[int], scores: dict[int, float]) -> list[dict]:
    """Milvus 命中 + 精排顺序 → 统一 entry 字典（chat 路由与 build_context 共用）。"""
    return [{"text": h["entity"]["text"], "content": h["entity"]["content"],
             "type": h["entity"]["chunk_type"], "page": h["entity"]["page"],
             "source": h["entity"]["source"], "rrf": h["distance"], "score": scores.get(i)}
            for i in order for h in [hits[i]]]


def retrieve(question: str, top_k: int = TOP_K):
    """混合检索：两路召回 + RRF 融合 Top-20 → rerank 精排 Top-3。
    rerank 失败降级回融合顺序（精度损失可控、可用性优先）。返回 (entries, reranked)。"""
    query_vec = embedder.embed_batch([question])[0]
    hits = store.hybrid_search(question, query_vec, top_k=RECALL_K)
    docs = [h["entity"]["text"] for h in hits]
    try:
        results = reranker.rerank(question, docs, top_n=top_k)
        order = [r["index"] for r in results]
        return _to_entries(hits, order, {r["index"]: r["relevance_score"] for r in results}), True
    except Exception:
        return _to_entries(hits, list(range(top_k)), {}), False


def build_context(retrieved: list[dict]) -> str:
    """把精排后的 Top-K 拼成带编号 + 类型标注的 context（回填 content 原文：表格=HTML、图片=描述）。"""
    parts = [f"[{i}][{llm._TYPE_LABEL.get(e['type'], '正文')}] {e['content'] or e['text']}"
             for i, e in enumerate(retrieved, 1)]
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
    chunks = splitter.split_text(text)
    rows = [{"text": c, "chunk_type": "text", "content": c, "page": 0} for c in chunks]
    store.insert_rows(rows, embedder.embed_batch(chunks), source)
    return len(chunks), True, f"[OK]「{source}」+{len(chunks)} 块"


def ingest_pdf(data: bytes, source: str) -> tuple[int, bool, str]:
    """多模态分支（PDF）：解析 → 向量化 → 入库。幂等同上。"""
    if store.has_source(source):
        return 0, False, f"[SKIP]「{source}」已在库中"
    rows, vlm_ok = parse_pdf(data, source, llm._ZHIPU_API_KEY, splitter)
    if not rows:
        return 0, True, f"[WARN]「{source}」未解析出内容（扫描件？本服务只吃有文字层的 PDF）"
    store.insert_rows(rows, embedder.embed_batch([r["text"] for r in rows]), source)
    by_type = {t: sum(1 for r in rows if r["chunk_type"] == t) for t in ("text", "table", "image")}
    imgs = by_type["image"]
    vlm_note = f"，VLM 描述 {vlm_ok}/{imgs}" if imgs else ""
    labels = {"text": "文本", "table": "表格", "image": "图片"}
    types = " + ".join(f"{v}{labels[k]}" for k, v in by_type.items() if v)
    return len(rows), True, f"[OK]「{source}」+{len(rows)} 块（{types}{vlm_note}）"
