"""存储与解析层：Milvus Lite 向量库 + 递归切分 + PDF 多模态解析

Milvus Lite 单文件库（data/mini_rag.db）：一张 collection 装下「原文 + 向量 + 稀疏 + 元信息」，
稠密（HNSW/COSINE）+ BM25（jieba 分析器）两路检索与 RRF 融合都在库内完成。
单进程文件锁：同一 data 目录同一时刻只允许一个进程打开（部署件别开多实例）。
"""

import hashlib
import io
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF：PDF 解析（文本块/表格/图片）
import pandas as pd
from docx import Document  # python-docx：Word 解析（段落/表格/内嵌图片）
from docx.table import Table
from docx.text.paragraph import Paragraph
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from .llm import describe_image

DATA_DIR = Path(__file__).parent.parent / "data"      # 向量库 + 图片原件 + 各种 sqlite
MEDIA_DIR = DATA_DIR / "media"                        # 图片原件（chunk 只存文件名，前端经 /api/media/ 回显）


@dataclass
class Chunk:
    """统一 chunk 数据模型：入库 / 检索 / 展示全链共用这一个定义（vector 由
    insert_rows 补齐，不在模型里）。之前 store 层叫 chunk_type、pipeline 转手
    改名 type、三个消费方各摸各的键——双名靠约定不散架，2026-10-30 收口。"""

    text: str            # 可检索文本：embedding 与 BM25 的输入
    chunk_type: str      # text | table | image
    content: str         # 命中时回填 LLM 的原文（表格=HTML；正文与 text 同）
    page: int            # PDF 页码；非 PDF / 跨页正文 = 0
    section: str = ""    # 章节名（standalone 表格/图片块用；正文跨章节不硬造）
    source: str = ""     # 来源文件名 stem（幂等键 + 展示 + 归因锚；insert_rows 统一补）

RECALL_K = 20     # 粗排宽进捞多少
TOP_K = 5         # 精排收紧留几条（进生成 context 的条数；10-29 多文档后 3→5：
                  # 评测两轮 命中@5 58.8% vs @3 47.1%，检索稳定 +11.7pp）
RRF_K = 60        # RRFRanker 的 k

# 章节锚（10-30 A/B 拍板）：独立表格块 text 加「source｜章节」前缀——22 条 × 3 轮
# 两臂对比：命中@3/@5、答对、拒绝全部持平（75/80/80/100 vs 75/80/80/100），
# Top-1 -5pp 在两臂重建库的 VLM 方差带内=中性 → 前缀默认不加（source 锚已吃掉
# 表格题检索增益）。section 保留为元数据（bot/web 脚注展示「哪份文档哪一节」）。
# RAGBOT_SECTION_ANCHOR=1 供语料变化后复验。
SECTION_ANCHOR = os.environ.get("RAGBOT_SECTION_ANCHOR", "0") == "1"

# 年报/公文风格章节标题：中文序号（一、/（一））或数字序号（1. / 2.1、），行短。
# 误报形态（重要提示里的「一、二、」列举项）会把 section 名带偏——启发式接受，
# 章节名只是检索辅助锚，不作结构承诺。
_SECTION_RE = re.compile(
    r"^(?:[（(]?[一二三四五六七八九十百]{1,4}[）)]?、?|\d+(?:\.\d+)*[、.．])\s*\S{1,28}$")


def _anchor(text: str, source: str, section: str) -> str:
    """独立表格块的可检索文本前缀：source 锚（必有）+ 章节锚（有则带上）。"""
    if SECTION_ANCHOR and section:
        return f"{source}｜{section}：{text}"
    return f"{source}：{text}"


class RecursiveTextSplitter:
    # 按优先级从高到低排列的分隔符：段落 → 换行 → 句号 → … → 空格 → 空串（字符级兜底，必须放末尾）
    SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")

    def __init__(self, chunk_size=200, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        pieces = self._split(text, self.SEPARATORS)
        return self._merge(pieces)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        for i, sep in enumerate(separators):
            if sep == "":
                return [text[j:j + self.chunk_size] for j in range(0, len(text), self.chunk_size)]
            if sep in text:
                rest = separators[i + 1:]
                break
        parts = text.split(sep)
        pieces = []
        for k, p in enumerate(parts):
            piece = p if k == len(parts) - 1 else p + sep     # 末片不焊本层 sep（防脏尾巴堆积）
            if len(piece) <= self.chunk_size:
                pieces.append(piece)
            else:
                pieces.extend(self._split(piece, rest))
        return pieces

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks, cur, cur_len = [], [], 0
        for p in pieces:
            if cur and cur_len + len(p) > self.chunk_size:
                chunks.append("".join(cur))
                while cur and cur_len > self.chunk_overlap:   # overlap：尾巴留给下一块
                    cur_len -= len(cur[0])
                    cur.pop(0)
            cur.append(p)
            cur_len += len(p)
        if cur:
            chunks.append("".join(cur))
        return chunks


class MilvusStore:
    """Milvus Lite 向量库：两路召回 + RRF 融合在库内完成，落库即持久化。"""

    COLLECTION = "docs"
    DIM = 1024                  # 与 llm.ZhipuEmbedder 的 dimensions 一致

    def __init__(self, db_path: str):
        # 传本地 .db 路径 = 启动嵌入式 Milvus Lite（不用装服务；文件存在则接着用）
        self.client = MilvusClient(db_path)
        self._ensure_collection()

    def _ensure_collection(self):
        """首次运行建 collection（schema + 索引）；已存在则加载后直接用。
        旧版（纯文本）schema 缺多模态字段 → drop 重建（重传语料即可）。"""
        if self.client.has_collection(self.COLLECTION):
            fields = {f["name"] for f in self.client.describe_collection(self.COLLECTION)["fields"]}
            if "chunk_type" not in fields or "section" not in fields:
                self.client.drop_collection(self.COLLECTION)   # 旧 schema（缺多模态/章节字段）→ 重灌
            else:
                self.client.load_collection(self.COLLECTION)
                return
        schema = MilvusClient.create_schema(auto_id=True)            # 主键由库自动分配
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field(
            "text", DataType.VARCHAR, max_length=65535,
            enable_analyzer=True, analyzer_params={"type": "jieba"},  # 中文按词切（不按字切）
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)      # BM25 函数的输出位，不用手填
        schema.add_function(Function(
            name="bm25", function_type=FunctionType.BM25,
            input_field_names=["text"], output_field_names=["sparse"],
        ))
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.DIM)
        schema.add_field("source", DataType.VARCHAR, max_length=256)  # 来源文件名（幂等键 + 展示）
        # —— 多模态三字段：chunk 不再只是纯文本 ——
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=8)   # text | table | image
        schema.add_field("content", DataType.VARCHAR, max_length=65535)  # 命中时回填给 LLM 的原文（表格=HTML）
        schema.add_field("page", DataType.INT64)                          # PDF 页码（非 PDF 语料 = 0）
        schema.add_field("section", DataType.VARCHAR, max_length=128)     # 章节名（standalone 块；正文跨章节为空）

        index = self.client.prepare_index_params()
        index.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                        params={"M": 16, "efConstruction": 200})
        index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                        metric_type="BM25")
        self.client.create_collection(self.COLLECTION, schema=schema, index_params=index)
        self.client.load_collection(self.COLLECTION)

    def insert_rows(self, chunks: list[Chunk], vectors: list[list[float]], source: str):
        """入库即持久化（库内 WAL 落盘）。source 统一在这里补进每一行（幂等键 + 展示）。"""
        self.client.insert(self.COLLECTION,
                           [{**asdict(c), "source": source, "vector": v}
                            for c, v in zip(chunks, vectors)])

    def has_source(self, source: str) -> bool:
        """幂等判断：该来源是否已在库（标量过滤查询）。"""
        rows = self.client.query(self.COLLECTION, filter=f'source == "{source}"',
                                 output_fields=["id"])
        return bool(rows)

    def hybrid_search(self, query: str, query_vec: list[float], top_k: int):
        """两路召回 + RRF 融合，库内一步完成。返回命中列表（含 text/source/chunk_type/content/page）。"""
        dense = AnnSearchRequest(          # 稠密路：语义相近
            data=[query_vec], anns_field="vector",
            param={"metric_type": "COSINE"}, limit=top_k)
        sparse = AnnSearchRequest(         # 稀疏路：字面命中（查询原文进来，库内 jieba 分词打 BM25）
            data=[query], anns_field="sparse",
            param={"metric_type": "BM25"}, limit=top_k)
        hits = self.client.hybrid_search(
            self.COLLECTION, [dense, sparse], ranker=RRFRanker(RRF_K),
            limit=top_k,
            output_fields=["text", "source", "chunk_type", "content", "page", "section"])
        return hits[0]

    def chunk_count(self) -> int:
        return int(self.client.get_collection_stats(self.COLLECTION)["row_count"])

    def sources(self) -> list[str]:
        rows = self.client.query(self.COLLECTION, filter='source != ""',
                                 output_fields=["source"], limit=16384)
        return sorted({r["source"] for r in rows})

    def clear(self):
        """清空：drop 掉 collection 重建——比删 .db 文件干净，也避开文件锁。"""
        self.client.drop_collection(self.COLLECTION)
        self._ensure_collection()


# ================================================================
# PDF 多模态解析管线：正文 / 表格 / 图片三种模态进同一检索空间
#   · 每页抽三类元素（文本块 / 表格 / 图片占位），按 y 坐标排序还原「元素流」，
#     表格区内的文本块剔除（表格整体转 HTML，不重复计字）
#   · 表格：HTML 保行列结构给 LLM 回填；文本化（表头+逐行拼接）给 embedding/BM25。
#     小表格（≤10 行）用边界标记跟正文一起切、上下文不丢；大表格独立成块，正文留引导句
#   · 图片：过滤装饰（短边过小 / 整页背景 / 同位重复 / 跨页同图），原件落 media/，
#     VLM 生成描述——描述是纯文本，独立成块参与检索，命中时前端回显原图
# ================================================================
HEADER_Y = 80          # 页眉区（顶栏公司名/年份，y1 低于此的文本块丢弃）
FOOTER_Y = 60          # 页脚区（页码，y0 高于页高-此值的文本块丢弃）
SMALL_TABLE_ROWS = 10  # ≤此行数的表格跟正文一起切；更大的独立成块
MIN_IMG_PT = 50        # 图片最短边（pt）低于此 = 装饰线/小图标，不入库
IMG_PAGE_RATIO = 0.8   # 单图占页面面积超过此比例 = 整页背景，不入库


def _overlap_ratio(rect: tuple, bbox: tuple) -> float:
    """rect 与 bbox 的交集面积占 rect 面积的比例（判定文本块是否落在表格里）。"""
    x0, y0 = max(rect[0], bbox[0]), max(rect[1], bbox[1])
    x1, y1 = min(rect[2], bbox[2]), min(rect[3], bbox[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = max((rect[2] - rect[0]) * (rect[3] - rect[1]), 1e-6)
    return (x1 - x0) * (y1 - y0) / area


def _page_elements(page) -> list[dict]:
    """一页 → 按 y 坐标排序的元素流 [{kind: text|table|image, y, ...}]。
    页眉页脚剔除；表格区内的文本块剔除；装饰图片剔除。"""
    tabs = page.find_tables().tables
    images, placed = [], []
    for x in page.get_images(full=True):
        for r in page.get_image_rects(x):
            if min(r.width, r.height) < MIN_IMG_PT:
                continue                            # 侧边装饰线（2pt 宽那种）
            if r.width * r.height > IMG_PAGE_RATIO * page.rect.width * page.rect.height:
                continue                            # 整页背景图
            if any(abs(r.x0 - p.x0) < 2 and abs(r.y0 - p.y0) < 2 for p in placed):
                continue                            # 同位叠放的双层图（阴影/遮罩），只取一个
            placed.append(r)
            images.append({"kind": "image", "y": r.y0, "xref": x[0]})
            break
    elements = []
    for x0, y0, x1, y1, txt, _no, btype in page.get_text("blocks"):
        if btype != 0 or not txt.strip():
            continue
        if y1 < HEADER_Y or y0 > page.rect.height - FOOTER_Y:
            continue                                # 页眉（公司名/年份）/ 页脚（页码）
        if any(_overlap_ratio((x0, y0, x1, y1), t.bbox) > 0.5 for t in tabs):
            continue                                # 表格区内的字——表格整体走 HTML，这里不重复计
        text = re.sub(r"[\r\n]+", "", txt)          # 块内硬换行是排版折行，拼回连续文本
        elements.append({"kind": "text", "y": y0, "text": text.strip()})
    elements.extend({"kind": "table", "y": t.bbox[1], "table": t} for t in tabs)
    elements.extend(images)
    elements.sort(key=lambda e: e["y"])
    return elements


def _table_parts(tab) -> tuple[str, str]:
    """PDF 表格 → (html, text)：PyMuPDF 表格先转 DataFrame（多级表头拉平、
    Col0/Col1 占位清空），再走与 docx 表格共用的 _df_parts。"""
    df = tab.to_pandas().fillna("")
    if isinstance(df.columns, pd.MultiIndex):       # 多级表头拉平成一行
        df.columns = [" ".join(map(str, c)) for c in df.columns]
    df.rename(columns=lambda c: "" if re.fullmatch(r"Col\d+", str(c)) else c,
              inplace=True)                         # PyMuPDF 没认出的表头占位（Col0/Col1…）清成空
    return _df_parts(df)


def _df_parts(df: pd.DataFrame) -> tuple[str, str]:
    """DataFrame → (html, text)：HTML 保行列结构（LLM 读表格远好于线性化文本），
    text 是文本化版（表头一行 + 逐行竖线拼接），给 embedding 和 BM25 用。"""
    html = df.to_html(index=False)
    lines = ["表头：" + "、".join(str(c) for c in df.columns)]
    lines += ["｜".join(str(v) for v in row) for row in df.itertuples(index=False)]
    return html, "\n".join(lines)


def _refill_tables(chunk: str, registry: dict[str, str]) -> str:
    """把 chunk 里的〔表格Tn〕…〔/表格Tn〕文本化内容替换回 HTML 原文（content 版）。
    被切开的残缺标记替换不到——降级保留文本版：信息不丢，只是没还原成表格。"""
    return re.sub(r"〔表格T(\d+)〕.*?〔/表格T\1〕",
                  lambda m: registry.get(m.group(1), m.group(0)), chunk, flags=re.S)


_TABLE_MARK = re.compile(r"〔/?表格T?\d*〕")        # text 版里剥掉边界标记（embedding 不需要）


def parse_pdf(data: bytes, source: str, vlm_key: str, splitter: RecursiveTextSplitter):
    """PDF 字节 → 多模态 chunk 行 [{text, chunk_type, content, page}...] + VLM 成功张数。
    三类产物：正文+小表格（跟正文一起切，text 型）、大表格（独立 table 型）、图片（独立 image 型）。"""
    doc = fitz.open(stream=data, filetype="pdf")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    stream, tables_html, standalone = [], {}, []
    seen_md5, vlm_ok = set(), 0
    section, tid = "", 0
    for pno, page in enumerate(doc, 1):
        page_parts = []
        for el in _page_elements(page):
            if el["kind"] == "text":
                if len(el["text"]) <= 30 and _SECTION_RE.match(el["text"]):
                    section = el["text"]           # 章节标题块：更新当前章节（照常入正文流）
                page_parts.append(el["text"])
            elif el["kind"] == "table":
                html, text = _table_parts(el["table"])
                n_rows = len(el["table"].rows)
                if n_rows <= SMALL_TABLE_ROWS:      # 小表格：边界标记内联，跟正文一起切
                    page_parts.append(f"〔表格T{tid}〕\n{text}\n〔/表格T{tid}〕")
                    tables_html[str(tid)] = html
                else:                               # 大表格：正文留引导句，HTML 独立成块
                    page_parts.append(f"〔大表格：{text.splitlines()[0]}，共{n_rows}行，独立成块〕")
                    # text 版带锚前缀：独立表格块脱离了正文上下文，没公司名/章节锚
                    # BM25/dense 两路都吃不到（10-29 M05 实证：星澜表格块被年报表挤出 Top-5）
                    standalone.append(Chunk(text=_anchor(text, source, section),
                                            chunk_type="table", content=html,
                                            page=pno, section=section))
                tid += 1
            else:                                   # 图片：VLM 描述独立成块，正文留引导句
                info = doc.extract_image(el["xref"])
                md5 = hashlib.md5(info["image"]).hexdigest()
                if md5 in seen_md5:                 # 跨页重复的同一张图，只入一次
                    continue
                seen_md5.add(md5)
                fname = f"{source}_p{pno}_{el['xref']}.{info['ext']}"
                (MEDIA_DIR / fname).write_bytes(info["image"])
                mime = f"image/{'jpeg' if info['ext'] in ('jpg', 'jpeg') else info['ext']}"
                desc = describe_image(vlm_key, info["image"], mime)
                if desc:
                    vlm_ok += 1
                else:
                    desc = "（图片，VLM 描述生成失败）"
                page_parts.append(f"〔图：{desc[:40]}…〕")
                standalone.append(Chunk(text=f"图片描述：{desc}", chunk_type="image",
                                        # 标记行不定界（无括号包裹）：文件名可含全角括号等字符，
                                        # 按行尾提取永不截断——用成对括号当定界符会被文件名里的括号截断
                                        content=f"{desc}\n原图文件：{fname}", page=pno))
        if page_parts:
            stream.append("\n\n".join(page_parts))
    # 正文流切块：text 版剥标记、content 版还原 HTML；正文 chunk 可能跨页，page 记 0（不标注）
    rows = standalone
    for chunk in splitter.split_text("\n\n".join(stream)):
        rows.append(Chunk(text=_TABLE_MARK.sub("", chunk),
                          chunk_type="text",
                          content=_refill_tables(chunk, tables_html),
                          page=0))
    return rows, vlm_ok


# ================================================================
# Word（.docx）解析管线：与 PDF 管线同构的三类 chunk
#   python-docx 的 paragraphs / tables 是两个平铺列表，阅读顺序只在 body 元素流里
#   ——按 body 迭代保持段/表交错序。表格共用 _df_parts（小表内联〔表格Tn〕边界
#   标记、大表独立成块）；内嵌图片 md5 去重后落 media/ + VLM 描述。
#   旧版 .doc（OLE2 二进制）python-docx 不吃——上传处明确引导转存 .docx。
# ================================================================
_NS_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"       # a:blip
_NS_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"  # r:embed


def _iter_block_items(doc):
    """按文档顺序产出 Paragraph / Table（python-docx 官方 recipe 的最简版）。"""
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def _docx_images(block) -> list[tuple[bytes, str]]:
    """块内内嵌图片 [(bytes, ext)…]——a:blip 的 r:embed 指到包内图片 part。"""
    out = []
    for blip in block._element.findall(f".//{_NS_A}blip"):
        part = block.part.related_parts.get(blip.get(f"{_NS_R}embed"))
        if part is not None and getattr(part, "blob", None):
            out.append((part.blob, part.partname.ext.lstrip(".") or "png"))
    return out


def parse_docx(data: bytes, source: str, vlm_key: str, splitter: RecursiveTextSplitter):
    """Word 文档字节 → 多模态 chunk 行，与 parse_pdf 同构（正文流/表格/图片三类）。
    docx 无页码：page 一律 0。返回 (rows, vlm_ok)。"""
    doc = Document(io.BytesIO(data))
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    parts, tables_html, standalone = [], {}, []
    seen_md5, vlm_ok, tid, img_n, section = set(), 0, 0, 0, ""
    for block in _iter_block_items(doc):
        for img_bytes, ext in _docx_images(block):
            md5 = hashlib.md5(img_bytes).hexdigest()
            if md5 in seen_md5:                    # 同图多处引用只入一次
                continue
            seen_md5.add(md5)
            img_n += 1
            fname = f"{source}_img{img_n}.{ext}"
            (MEDIA_DIR / fname).write_bytes(img_bytes)
            mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
            desc = describe_image(vlm_key, img_bytes, mime)
            if desc:
                vlm_ok += 1
            else:
                desc = "（图片，VLM 描述生成失败）"
            parts.append(f"〔图：{desc[:40]}…〕")
            standalone.append(Chunk(text=f"图片描述：{desc}", chunk_type="image",
                                    # 标记行不定界：与 PDF 侧同口径（main.py 按行尾提取文件名）
                                    content=f"{desc}\n原图文件：{fname}", page=0))
        if isinstance(block, Table):
            grid = [[c.text.strip() for c in r.cells] for r in block.rows]
            if not grid:                           # 空表（python-docx 偶见占位表）
                continue
            html, text = _df_parts(pd.DataFrame(grid[1:], columns=grid[0]))
            n_rows = len(block.rows)
            if n_rows <= SMALL_TABLE_ROWS:         # 小表格：边界标记内联，跟正文一起切
                parts.append(f"〔表格T{tid}〕\n{text}\n〔/表格T{tid}〕")
                tables_html[str(tid)] = html
            else:                                  # 大表格：正文留引导句，HTML 独立成块
                parts.append(f"〔大表格：{text.splitlines()[0]}，共{n_rows}行，独立成块〕")
                standalone.append(Chunk(text=_anchor(text, source, section),
                                        chunk_type="table", content=html,
                                        page=0, section=section))
            tid += 1
        elif isinstance(block, Paragraph) and block.text.strip():
            txt = block.text.strip()
            # 章节标题两路识别：Heading 样式优先（规范文档）；兜底=加粗+序号正则
            # （真实世界 docx 常用加粗大字号伪装标题——万华反样式样本专验这条路）
            style = block.style.name if block.style is not None else ""
            if style.startswith("Heading") or style.startswith("标题") or (
                    len(txt) <= 30 and _SECTION_RE.match(txt)
                    and any(r.bold for r in block.runs if r.text.strip())):
                section = txt
            parts.append(txt)
    rows = standalone
    for chunk in splitter.split_text("\n\n".join(parts)):
        rows.append(Chunk(text=_TABLE_MARK.sub("", chunk),
                          chunk_type="text",
                          content=_refill_tables(chunk, tables_html),
                          page=0))
    return rows, vlm_ok
