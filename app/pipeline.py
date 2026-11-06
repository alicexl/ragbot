"""RAG 管线胶合层：单例接线 + 检索/增强/生成入口

单例（embedder / splitter / store / reranker / generator）由 init() 在入口接好线，
web 路由、钉钉 bot、评测脚本三个调用方共享同一套——Milvus 单进程锁决定了
全服务必须单进程，模块级单例 + 调用时属性访问是最直白的形态。

检索链路：pre-retrieval（PRE_MODE，可选）→ embed → Milvus 两路召回 + RRF 融合 Top-20
→ rerank 精排 Top-3（失败降级融合序）。生成拼装口径（【参考资料】+【用户问题】）
全调用方同构——评测测的就是线上这条链。
"""

import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass

from . import llm, trace
from .store import (DATA_DIR, MEDIA_DIR, RECALL_K, TOP_K, Chunk, DocRegistry,
                    MilvusStore, RecursiveTextSplitter, parse_docx, parse_pdf)


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
registry = None
reranker = None
generator = None


def init(db_path: str | None = None):
    """入口调用：配置检查 → 初始化全部单例。返回 None（失败直接抛 SystemExit）。
    db_path 不传 = 生产库 data/mini_rag.db；评测传 eval.db，与生产数据隔离。"""
    global embedder, splitter, store, registry, reranker, generator
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
    db = db_path or str(DATA_DIR / "mini_rag.db")
    store = MilvusStore(db)
    registry = DocRegistry(db)          # 文档注册表随库文件隔离（mini_rag_docs.db / eval_docs.db）


def _to_entries(hits: list[dict], order: list[int], scores: dict[int, float]) -> list[RetrievedChunk]:
    """Milvus 命中 + 精排顺序 → RetrievedChunk（chat 路由与 build_context 共用）。"""
    return [RetrievedChunk(text=h["entity"]["text"], content=h["entity"]["content"],
                           chunk_type=h["entity"]["chunk_type"], page=h["entity"]["page"],
                           section=h["entity"].get("section", ""),
                           source=h["entity"]["source"],
                           rrf=h["distance"], score=scores.get(i))
            for i in order for h in [hits[i]]]


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def source_lock(source: str) -> threading.Lock:
    """同 source 的 ingest/delete 串行化（11-02 实证修复：与页面并发上传同一文档
    → 两请求都过 has_source 检查 → 双入库，而 md5 守卫对「同内容并存」是 SKIP
    不自愈）。按文档粒度、进程内——服务单进程，跨进程由 Milvus 文件锁兜底；
    锁只增不删（每文档一把，量级=文档数，不值得回收）。"""
    with _locks_guard:
        return _locks.setdefault(source, threading.Lock())


def detect_entities(question: str) -> list[tuple[str, list[str]]]:
    """问句命中的库内实体（when/who/what 里的 who）及其全部 source，长名优先防
    子串误配。词典来自 sources() 去掉「20XX年报/简报」年份后缀的公司名——库内
    实体是有限集合，字符串命中即高置信，规则匹配不走 LLM（与斜杠命令走规则同一
    哲学）。实体用于圈定检索范围（标量过滤），不参与排序——排序交给残句里的
    when（年份）/what（指标），否则实体名高词频块（释义表）会碾压指标块。"""
    by_entity: dict[str, list[str]] = {}
    for src in store.sources():
        m = re.match(r"^(.+?)20\d{2}", src)
        by_entity.setdefault(m.group(1) if m else src, []).append(src)
    return [(n, by_entity[n]) for n in sorted(by_entity, key=len, reverse=True)
            if n in question]


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

    queries = [question]      # BM25 路用的查询文本（multi/per-entity 时为列表）
    vec_texts = [question]    # dense 路要向量化的文本（hyde 时为假设文档）
    filters = [""]            # 各路标量过滤（per-entity 路按公司圈定，其余路不过滤）
    entities = detect_entities(question) if mode == "off" else []
    if len(entities) >= 2:
        # per-entity fan-out（多实体问题的 when/who/what 拆解）：who 用于**圈定范围**
        # （source in [该公司全部文档]）而非参与排序——实体名留在查询里会让释义表
        # 这类高词频块霸榜（云天化三份释义表霸掉全部坑位，p4 会计表进不了 context）；
        # 排序交给残句的 when/what（年份+指标），公司内部「2024 营业收入」自然顶到
        # p4 表。各实体路按名次轮转占坑（每实体第一名必进 context），余位全局 RRF
        # 补齐（原问路只补坑不占坑）；单实体问题不拆，零开销。与 pre 模式互斥不组合。
        stripped = question
        for e, _ in entities:
            stripped = stripped.replace(e, " ")
        queries = vec_texts = [stripped] * len(entities) + [question]
        filters = ['source in [' + ",".join(f'"{s}"' for s in srcs) + ']'
                   for _, srcs in entities] + [""]
        trace.record(rid, "entity_fanout", 0.0,
                     meta={"entities": [e for e, _ in entities]})
    elif mode != "off":
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
    elif entities:
        # per-entity 轮转配额合并：各实体路按名次轮转先占坑——每实体的第一名必进
        # context。纯 RRF 会让跨路噪声挤掉单路第一名（M01 实测：云天化路 rank-1 被
        # distractor 挤出 Top-5，模型对着缺块的 context 如实拒答）；不足 top_k 再按
        # 全局 RRF 补齐，原问路不占坑只参与补齐。
        route_lists = [store.hybrid_search(q, v, top_k=RECALL_K, filter=f)
                       for q, v, f in zip(queries, vecs, filters)]
        hits, seen = [], set()
        for pair in zip(*route_lists[:-1]):            # 末路 = 原问，不占坑
            for h in pair:
                if h["id"] not in seen and len(hits) < top_k:
                    seen.add(h["id"])
                    hits.append(h)
        scores: dict[int, float] = {}
        by_id: dict[int, dict] = {}
        for route in route_lists:
            for rank, h in enumerate(route):
                hid = h["id"]
                scores[hid] = scores.get(hid, 0.0) + 1.0 / (60 + rank)
                by_id.setdefault(hid, h)
        hits += [by_id[hid] for hid in sorted(scores, key=scores.get, reverse=True)
                 if hid not in seen][:top_k - len(hits)]
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


def group_by_source(entries: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """展示序收口：同 source 的块聚到一起（组序=该文档首个块的检索名次，组内保持
    名次序），(source, text) 全同的块去重留首个（上游已按 Milvus id 去重，这里兜底
    跨路同文本不同 id 的极端形态）。build_context 的引用编号与 web sources 面板 /
    bot 脚注的编号必须同源——三个消费方都从这里取展示序；retrieve() 的返回序
    （检索名次，eval 判分口径）不动。"""
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[RetrievedChunk]] = {}
    for e in entries:
        key = (e.source, e.content or e.text)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(e.source, []).append(e)
    return [e for chunks in grouped.values() for e in chunks]


def build_context(retrieved: list[RetrievedChunk]) -> str:
    """Top-K 按文档分组拼装：【文档：X】头 + 组内 [n][类型] + content（回填原文：
    表格=HTML、图片=描述），编号跨组连续。分组给多文档题（M01/M02 跨公司比较）
    「每文档一份」的清晰结构；来源标注让模型按公司/年份正确归因——10-29 实测：
    无来源标注时模型把盐湖的数字安到云天化头上（M01/M03 探针）。"""
    parts: list[str] = []
    src = None
    for i, e in enumerate(group_by_source(retrieved), 1):
        if e.source != src:
            src = e.source
            parts.append(f"【文档：{e.source}】")
        parts.append(f"[{i}][{llm._TYPE_LABEL.get(e.chunk_type, '正文')}] {e.content or e.text}")
    return "\n".join(parts)


def build_user_msg(question: str, retrieved: list[dict], history: list[dict] | None = None) -> list[dict]:
    """生成的 messages 拼装（全调用方同构）：多轮 history + 本轮带参考资料的问题。"""
    context = build_context(retrieved)
    user_msg = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"
    return (history or []) + [{"role": "user", "content": user_msg}]


def ingest_text(text: str, source: str, replace: bool = False) -> tuple[int, bool, str]:
    """切块 + 向量化 + 入库一份纯文本文档。幂等与覆盖：source 已在库且未勾覆盖则跳过；
    覆盖时 md5 相同=内容未变化直接跳过，不同走文档级替换（见 _ingest_rows）。
    全程持 per-source 锁——检查与写入原子化，并发同文档上传串行成 一存一 SKIP。"""
    with source_lock(source):
        old_md5 = store.source_md5(source)
        if old_md5 is not None and not replace:
            return 0, False, f"[SKIP]「{source}」已在库中（勾选覆盖重传可更新）"
        new_md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        if old_md5 == new_md5:
            return 0, False, f"[SKIP]「{source}」已在库中且内容未变化"
        texts = splitter.split_text(text)
        chunks = [Chunk(text=c, chunk_type="text", content=c, page=0) for c in texts]
        return _ingest_rows(source, chunks, 0, new_md5, old_md5 is not None)


def _ingest_rows(source: str, chunks: list[Chunk], vlm_ok: int, md5: str,
                 replacing: bool) -> tuple[int, bool, str]:
    """多模态 chunk 入库 + ingest span + 结果摘要（三个 ingest 共用尾巴）。
    replacing=True（覆盖更新）：**先传后删**——insert 前记下旧块主键、insert 成功后再删。
    旧版保留到新版确定入座，parse/embed/insert 任一步失败库内容不丢；失败窗口的
    新旧并存由下次覆盖自愈（重查该 source 全量 ID 一并删掉）。"""
    rid = trace.new_request_id()
    t0 = time.perf_counter()
    old_ids = store.source_ids(source) if replacing else []
    store.insert_rows(chunks, embedder.embed_batch([c.text for c in chunks]), source, md5)
    store.delete_ids(old_ids)
    by_type = {t: sum(1 for c in chunks if c.chunk_type == t) for t in ("text", "table", "image")}
    imgs = by_type["image"]
    vlm_note = f"，VLM 描述 {vlm_ok}/{imgs}" if imgs else ""
    labels = {"text": "文本", "table": "表格", "image": "图片"}
    types = " + ".join(f"{v}{labels[k]}" for k, v in by_type.items() if v)
    trace.record(rid, "ingest", (time.perf_counter() - t0) * 1000,
                 meta={"source": source, "chunks": len(chunks), "vlm_ok": vlm_ok,
                       "replaced": len(old_ids)})
    registry.upsert(source, len(chunks),
                    {"types": {k: v for k, v in by_type.items() if v}, "vlm_ok": vlm_ok})
    verb = "替换旧版 " if replacing else ""
    return len(chunks), True, f"[OK]「{source}」{verb}+{len(chunks)} 块（{types}{vlm_note}）"


def ingest_pdf(data: bytes, source: str, replace: bool = False) -> tuple[int, bool, str]:
    """多模态分支（PDF）：解析 → 向量化 → 入库。幂等与覆盖同 ingest_text（含 per-source 锁）。"""
    with source_lock(source):
        old_md5 = store.source_md5(source)
        if old_md5 is not None and not replace:
            return 0, False, f"[SKIP]「{source}」已在库中（勾选覆盖重传可更新）"
        new_md5 = hashlib.md5(data).hexdigest()
        if old_md5 == new_md5:
            return 0, False, f"[SKIP]「{source}」已在库中且内容未变化"
        rows, vlm_ok = parse_pdf(data, source, llm._ZHIPU_API_KEY, splitter)
        if not rows:
            return 0, True, f"[WARN]「{source}」未解析出内容（扫描件？本服务只吃有文字层的 PDF）"
        return _ingest_rows(source, rows, vlm_ok, new_md5, old_md5 is not None)


def ingest_docx(data: bytes, source: str, replace: bool = False) -> tuple[int, bool, str]:
    """Word 分支（.docx）：与 PDF 管线同构（段落/表格/图片三类 chunk）。幂等与覆盖同上。"""
    with source_lock(source):
        old_md5 = store.source_md5(source)
        if old_md5 is not None and not replace:
            return 0, False, f"[SKIP]「{source}」已在库中（勾选覆盖重传可更新）"
        new_md5 = hashlib.md5(data).hexdigest()
        if old_md5 == new_md5:
            return 0, False, f"[SKIP]「{source}」已在库中且内容未变化"
        rows, vlm_ok = parse_docx(data, source, llm._ZHIPU_API_KEY, splitter)
        if not rows:
            return 0, True, f"[WARN]「{source}」未解析出内容（空文档？）"
        return _ingest_rows(source, rows, vlm_ok, new_md5, old_md5 is not None)
