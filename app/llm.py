"""模型 API 客户端层：embedding / rerank / 生成 / VLM 四类调用与 key 拓扑

key 拓扑（拆分背景：生成商可独立切换，检索侧固定智谱）：
- ANTHROPIC_API_KEY → 智谱 Key：embedding-3 / rerank / VLM（glm-4v-flash）走智谱原生 API
- CHAT_API_KEY      → 生成专用：默认复用智谱 glm-5.3（anthropic 兼容网关），
  也可指向 DeepSeek（https://api.deepseek.com/anthropic + deepseek-flash），
  配 ANTHROPIC_BASE_URL / ANTHROPIC_CHAT_MODEL 两个 env 即完成切换
"""

import base64
import os

import requests
from anthropic import Anthropic

API_KEY = ""   # 没设 env 时的兜底（推荐环境变量，免改代码）

_CHAT_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
_CHAT_MODEL = os.environ.get("ANTHROPIC_CHAT_MODEL", "glm-5.3")
_CHAT_API_KEY = os.environ.get("CHAT_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or API_KEY
_ZHIPU_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or API_KEY

_ZHIPU_EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
_ZHIPU_EMBED_DIM = 1024        # 必须与 store 的 vector 维度一致（API 不指定默认 2048）
_RERANK_URL = "https://open.bigmodel.cn/api/paas/v4/rerank"
_VLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
_VLM_MODEL = os.environ.get("ZHIPU_VLM_MODEL", "glm-4v-flash")

SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。只能根据下面【参考资料】里的内容回答用户问题，"
    "并在用到某条资料时标注它的编号（如「根据[1]」）。"
    "参考资料带类型标注：[正文]是普通文本，[表格]是 HTML 表格（行列结构完整，按单元格读数），"
    "[图片]是对原图内容的文字描述。"
    "如果【参考资料】里没有相关信息，就如实回答「资料中未提及」，禁止编造。"
)

_TYPE_LABEL = {"text": "正文", "table": "表格", "image": "图片"}


class ZhipuEmbedder:
    """智谱 embedding-3：1024 维；检索链路（稠密路召回 + 入库向量化）都用它"""

    def __init__(self, api_key: str):
        self.model = "embedding-3"
        self.api_key = api_key

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            _ZHIPU_EMBED_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(texts), "dimensions": _ZHIPU_EMBED_DIM},
            timeout=60,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])   # 批量返回按 index 对齐
        return [d["embedding"] for d in data]


class Reranker:
    """智谱 rerank（Cross-Encoder 精排）。失败时调用方降级回融合顺序——精度损失可控"""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("rerank 需要智谱 API Key（ANTHROPIC_API_KEY）")
        self.api_key = api_key

    def rerank(self, query: str, documents: list[str], top_n: int = 3) -> list[dict]:
        """返回按相关性降序的 [{index, relevance_score}, ...]，index 指回 documents 的下标。"""
        resp = requests.post(
            _RERANK_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "rerank", "query": query, "documents": documents, "top_n": top_n},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]


class Generator:
    """流式生成（anthropic SDK，兼容智谱网关与 DeepSeek 的 anthropic 端点）"""

    def __init__(self, base_url: str = _CHAT_BASE_URL, model: str = _CHAT_MODEL):
        self.base_url = base_url
        self.model = model
        self.api_key = _CHAT_API_KEY

    def answer_stream(self, system: str, messages: list[dict], max_tokens: int = 2048,
                      thinking_off: bool = False):
        if not self.api_key:
            raise RuntimeError("未配置生成 API Key（CHAT_API_KEY / ANTHROPIC_API_KEY）")
        client = Anthropic(base_url=self.base_url, api_key=self.api_key)
        kw = {"thinking": {"type": "disabled"}} if thinking_off else {}
        with client.messages.stream(model=self.model, max_tokens=max_tokens,
                                    system=system, messages=messages, **kw) as stream:
            for text in stream.text_stream:
                yield text


def describe_image(api_key: str, img_bytes: bytes, mime: str) -> str | None:
    """VLM（glm-4v-flash）给图片生成可检索的描述文本。失败返回 None——
    调用方降级为占位描述，不阻塞入库（离线管道的容错原则）。"""
    b64 = base64.b64encode(img_bytes).decode()
    try:
        resp = requests.post(
            _VLM_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=60,
            json={"model": _VLM_MODEL, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": "这是公司年报里的图片。用一两句话客观描述它：图表类型、"
                                         "主体是什么、关键趋势或结论。只描述看得到的内容，不推测具体数字。"},
            ]}]},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
