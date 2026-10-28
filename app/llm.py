"""模型 API 客户端层：embedding / rerank / 生成 / VLM 四类调用与 key 拓扑

key 拓扑（拆分背景：生成商可独立切换，检索侧固定智谱）：
- ANTHROPIC_API_KEY → 智谱 Key：embedding-3 / rerank / VLM（glm-4v-flash）走智谱原生 API
- CHAT_API_KEY      → 生成专用：默认复用智谱 glm-5.3（anthropic 兼容网关），
  也可指向 DeepSeek（https://api.deepseek.com/anthropic + deepseek-flash），
  配 ANTHROPIC_BASE_URL / ANTHROPIC_CHAT_MODEL 两个 env 即完成切换
"""

import base64
import logging
import os
import time

import httpx
import requests
from anthropic import Anthropic

log = logging.getLogger("ragbot.llm")

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
            timeout=30,
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
        """返回按相关性降序的 [{index, relevance_score}, ...]，index 指回 documents 的下标。
        超时 8s 快速失败——调用方降级回融合顺序，不重试（降级本身就是它的容错）。"""
        resp = requests.post(
            _RERANK_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "rerank", "query": query, "documents": documents, "top_n": top_n},
            timeout=8,
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
    """VLM（glm-4v-flash）给图片生成可检索的描述文本。超时 20s、重试一次；
    仍失败返回 None——调用方降级为占位描述，不阻塞入库（离线管道的容错原则）。"""
    b64 = base64.b64encode(img_bytes).decode()
    try:
        return call_with_retry(
            lambda: _describe_image_once(api_key, b64, mime),
            attempts=2, backoff=1.0, what="vlm")
    except Exception as e:
        log.warning("VLM 描述失败（降级占位）：%s", e)
        return None


def _describe_image_once(api_key: str, b64: str, mime: str) -> str:
    resp = requests.post(
        _VLM_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=20,
        json={"model": _VLM_MODEL, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": "这是公司年报里的图片。用一两句话客观描述它：图表类型、"
                                     "主体是什么、关键趋势或结论。只描述看得到的内容，不推测具体数字。"},
        ]}]},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ================================================================
# 可靠性层：有限重试 + 流式生成的首字超时
# ================================================================
def call_with_retry(fn, *, attempts: int = 2, backoff: float = 0.8, what: str = ""):
    """有限重试（指数退避）。只包幂等的读类调用（embedding/VLM）；rerank 不重试
    （失败即降级，降级就是它的容错）；生成走 StreamedAnswer 的首字超时重试。"""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < attempts - 1:
                delay = backoff * (2 ** i)
                log.warning("%s 第 %d 次失败（%s），%.1fs 后重试", what, i + 1, e, delay)
                time.sleep(delay)
    raise last


# 流式生成的两个超时：read=首字（含流中任意两片的间隔）超时由 httpx 抛出；
# 总时长在迭代里手动查表。首字超时重试一次——长尾大多是排队抖动，重试大概率落快通道。
FIRST_TOKEN_TIMEOUT = 15.0
GEN_TOTAL_TIMEOUT = 90.0


class StreamedAnswer:
    """流式生成（Web SSE 与钉钉打字机共用）：
    - 首字/流间隔超时 15s（httpx read timeout）→ 取消重试一次，再失败上抛
    - 总时长 90s 兜底
    - 用法：sa = StreamedAnswer(...)；for delta in sa: ...；事后 sa.usage 取 token 数"""

    def __init__(self, system: str, messages: list[dict], rid: str | None = None,
                 thinking_off: bool = False, max_tokens: int = 2048):
        from . import trace
        self.system, self.messages, self.rid = system, messages, rid
        self.thinking_off, self.max_tokens = thinking_off, max_tokens
        self.usage = 0
        self._trace = trace

    def __iter__(self):
        client = Anthropic(base_url=_CHAT_BASE_URL, api_key=_CHAT_API_KEY,
                           timeout=httpx.Timeout(10.0, read=FIRST_TOKEN_TIMEOUT))
        kw = {"thinking": {"type": "disabled"}} if self.thinking_off else {}
        t0 = time.perf_counter()
        for attempt in (1, 2):
            buf = ""
            try:
                with client.messages.stream(model=_CHAT_MODEL, max_tokens=self.max_tokens,
                                            system=self.system, messages=self.messages, **kw) as s:
                    first = True
                    for delta in s.text_stream:
                        if first and delta.strip():
                            first = False
                            self._trace.record(self.rid, "gen_first_token",
                                               (time.perf_counter() - t0) * 1000,
                                               meta={"attempt": attempt} if attempt > 1 else None)
                        if time.perf_counter() - t0 > GEN_TOTAL_TIMEOUT:
                            raise TimeoutError(f"生成总时长超 {GEN_TOTAL_TIMEOUT}s")
                        buf += delta
                        yield delta
                    self.usage = (s.get_final_message().usage.input_tokens
                                  + s.get_final_message().usage.output_tokens)
                self._trace.record(self.rid, "gen_total", (time.perf_counter() - t0) * 1000,
                                   meta={"attempt": attempt} if attempt > 1 else None)
                return
            except TimeoutError:
                self._trace.record(self.rid, "gen_total", (time.perf_counter() - t0) * 1000,
                                   status="error", meta={"reason": "total_timeout"})
                raise
            except Exception as e:
                if attempt == 1:
                    log.warning("生成失败（%s），重试一次", e)
                    t0 = time.perf_counter()
                    continue
                self._trace.record(self.rid, "gen_total", (time.perf_counter() - t0) * 1000,
                                   status="error", meta={"error": str(e)[:120]})
                raise
