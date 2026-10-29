"""钉钉机器人（Stream 模式）：把 RAG 服务接进聊天工具

Stream 模式 = 出站 WebSocket：本进程主动连钉钉服务器，零公网 IP / 域名 / 证书。
与 Web 端的两处对照（聊天平台替你藏掉的东西，这里摊开）：
- 会话状态必须服务端持有（Web 端由前端上送）——SQLite 按 conversation_id，入史=原始问题
- history 表同时是使用日志（问题/答案/来源/延迟/token），坏例排查与真实 query 积累从这里长出来

回复链路：收到消息 → 先 ACK 再异步回答（生成长尾会触发钉钉超时重推，message_id 去重）
→ 先发 AI 卡片（「执行中」占位）→ 检索 → 流式生成边收边写卡片（打字机三要素：
首帧置 INPUTING / 中间帧底层直发 / 终帧 isFinalize 收尾，缺一不可）→ ai_finish 定稿。
生成关思考（数值问答不需要，快 ~30%）；usage 从流的 get_final_message() 拿。
卡片创建静默失败须显式校验 card_instance_id（SDK 不抛异常，废卡会吞掉降级路径）。

启动由 main() 自动接线（env：DINGTALK_CLIENT_ID / DINGTALK_CLIENT_SECRET 都设才起）。
权限：应用需开通 Card.Instance.Write（建卡）+ Card.Streaming.Write（流写）。
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
import time

import dingtalk_stream
from dingtalk_stream import AckMessage

from . import limits, llm, pipeline, trace
from .store import DATA_DIR, TOP_K

log = logging.getLogger("ragbot.bot")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [BOT] %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

# 会话历史：SQLite（服务端持有）；这张表同时是使用日志
HISTORY_DB = str(DATA_DIR / "bot_history.db")
HISTORY_WINDOW = 6          # 取最近 6 条消息（= 3 轮问答）拼生成的 messages
MAX_REPLY_CHARS = 3000      # 钉钉 markdown 单条上限远大于此，先截断保阅读体验

REFUSE_PATTERNS = ["未提及", "未涉及", "没有相关", "未提供", "资料中没"]


def _conn() -> sqlite3.Connection:
    """每次操作开短连接：SQLite 跨线程共用连接易锁，这个量级短连接最省心"""
    c = sqlite3.connect(HISTORY_DB)
    c.execute("""CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,             -- user 存原始问题（不带参考资料，防多轮滚雪球）
        content TEXT NOT NULL,
        sources TEXT,                   -- assistant 行专属：命中来源 JSON（同时是使用日志）
        latency_ms INTEGER,             -- assistant 行专属：检索+生成总耗时
        tokens INTEGER,                 -- assistant 行专属：input+output token
        ts REAL NOT NULL)""")
    return c


def load_history(conversation_id: str) -> list[dict]:
    """取该会话最近 N 条（旧→新）。群聊想按人隔离时把键换成 conversation_id+sender 即可"""
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? "
            "ORDER BY id DESC LIMIT ?", (conversation_id, HISTORY_WINDOW)).fetchall()
    return [{"role": r, "content": t} for r, t in reversed(rows)]


def append_history(conversation_id: str, role: str, content: str,
                   sources: str | None = None, latency_ms: int | None = None,
                   tokens: int | None = None) -> None:
    with _conn() as c:
        c.execute("INSERT INTO messages(conversation_id, role, content, sources, latency_ms, tokens, ts) "
                  "VALUES(?,?,?,?,?,?,?)",
                  (conversation_id, role, content, sources, latency_ms, tokens, time.time()))


def build_reply(answer: str, retrieved: list[dict]) -> str:
    """markdown 回复：答案 + 引用来源脚注（Web 端来源面板的机器人版）"""
    if len(answer) > MAX_REPLY_CHARS:
        answer = answer[:MAX_REPLY_CHARS] + "\n\n…（超长截断，完整答案见 Web 界面）"
    parts = [answer]
    if retrieved:
        parts.append("\n\n---\n**本次依据**")
        for i, e in enumerate(retrieved, 1):
            where = f" {e['source']} p{e['page']}" if e["page"] else f" {e['source']}"
            parts.append(f"\n[{i}]（{llm._TYPE_LABEL.get(e['type'], '正文')}）{where}")
    return "".join(parts)


def generate_stream(question: str, history: list[dict], retrieved: list[dict],
                    card, rid: str) -> tuple[str, int]:
    """流式生成 + AI 卡片打字机。生成走 llm.StreamedAnswer（首字超时重试/总时长兜底
    都在里面，与 Web 端同一条生成路径）；thinking 关闭：数值问答不需要长思考。
    卡片推送 ~1s 节流；usage 事后从 sa.usage 取。"""
    buf, last_push, n_push = "", 0.0, 0
    sa = llm.StreamedAnswer(llm.SYSTEM_PROMPT,
                            pipeline.build_user_msg(question, retrieved, history),
                            rid=rid, thinking_off=True)
    for delta in sa:
        buf += delta
        now = time.time()
        if card and now - last_push >= 1.0:
            # 打字机三要素：首帧经 ai_streaming 置 INPUTING 态，后续帧底层直发全量，
            # 终帧 isFinalize 由调用方补——SDK 的 ai_streaming 永远不置 finished，
            # 缺终帧时卡片会挂在流式会话里、只渲染第一帧
            if n_push == 0:
                card.ai_streaming(buf)
            else:
                card.streaming(card.card_instance_id, "msgContent", buf,
                               append=False, finished=False, failed=False)
            n_push += 1
            last_push = now
    log.info("流式 %d 帧 / 共 %d 字", n_push, len(buf))
    return buf, sa.usage


class RAGBotHandler(dingtalk_stream.ChatbotHandler):
    """钉钉消息 → RAG 链路 → 卡片流式回复。ACK 与回答分离：process 立刻返回，
    生成丢线程池异步做——同步等完长尾生成会触发钉钉超时重推。"""

    def __init__(self):
        super().__init__()
        self._seen: dict[str, float] = {}     # message_id -> ts，挡重推造成的重复回答
        self._lock = threading.Lock()

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        msg = dingtalk_stream.ChatbotMessage.from_dict(callback.data)
        with self._lock:
            stale = [k for k, ts in self._seen.items() if time.time() - ts > 600]
            for k in stale:
                self._seen.pop(k, None)
            if msg.message_id and msg.message_id in self._seen:
                return AckMessage.STATUS_OK, "OK"      # 重推消息：已答过，直接 ACK
            if msg.message_id:
                self._seen[msg.message_id] = time.time()
        asyncio.get_running_loop().create_task(self._answer(msg))
        return AckMessage.STATUS_OK, "OK"

    async def _answer(self, msg: dingtalk_stream.ChatbotMessage):
        loop = asyncio.get_running_loop()
        holder = {"card": None}                          # 异常时拿得到卡片，好调 ai_fail
        try:
            await loop.run_in_executor(None, self._answer_sync, msg, holder)
        except Exception as e:                          # 异步任务里的异常没人接，自己兜
            log.error("回答失败：%s", e)
            if holder["card"]:
                try:
                    holder["card"].ai_fail()
                except Exception:
                    pass
            try:
                self.reply_text(f"回答失败：{e}", msg)
            except Exception:
                pass

    def _answer_sync(self, msg: dingtalk_stream.ChatbotMessage, holder: dict):
        question = (msg.text.content if msg.text else "").strip()
        question = re.sub(r"^@\S+\s*", "", question)     # 群聊 @机器人 前缀剥掉
        nick = msg.sender_nick or msg.sender_staff_id or "?"
        user = f"dingtalk:{msg.sender_staff_id or nick}"
        conv = msg.conversation_id or "default"
        if not question:
            self.reply_text("暂时只支持文字提问。", msg)
            return
        if not pipeline.store.chunk_count():
            self.reply_text("向量库是空的：先在 Web 界面导入文档，再回来提问。", msg)
            return
        ok, why = limits.allow(user)
        if not ok:
            self.reply_text(why, msg)
            return
        limits.count_question(user)

        rid = trace.new_request_id()
        t0 = time.perf_counter()
        history = load_history(conv)
        retrieved, reranked = pipeline.retrieve(question, top_k=TOP_K, rid=rid,
                                                history=history)

        # AI 卡片先上（「执行中」占位）。注意 SDK 的坑：建卡失败是静默的（不抛异常、
        # card_instance_id 留空），必须显式校验，否则废卡会吞掉降级路径
        try:
            card = self.ai_markdown_card_start(msg, title="ragbot")
            holder["card"] = card if getattr(card, "card_instance_id", None) else None
            if card and holder["card"] is None:
                log.warning("AI 卡片创建静默失败（Card.Instance.Write 权限？），降级整条回复")
            log.info("卡片 %s", holder["card"].card_instance_id if holder["card"] else "降级整条回复")
        except Exception as e:
            log.warning("AI 卡片创建失败，降级整条回复：%s", e)

        answer, tokens = generate_stream(question, history, retrieved, holder["card"], rid)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        limits.count_tokens(tokens)
        trace.record(rid, "e2e", (time.perf_counter() - t0) * 1000, meta={"via": "dingtalk"})

        sources = json.dumps([{"i": i, "type": e["type"], "source": e["source"], "page": e["page"]}
                              for i, e in enumerate(retrieved, 1)], ensure_ascii=False)
        append_history(conv, "user", question)           # 入史的是原始问题（同 Web 端口径）
        append_history(conv, "assistant", answer, sources=sources,
                       latency_ms=latency_ms, tokens=tokens)
        reply_md = build_reply(answer, retrieved)
        if holder["card"]:
            c = holder["card"]
            # isFinalize 终帧把流式会话按协议收掉（含来源脚注全文），再 ai_finish 刷完成态
            c.streaming(c.card_instance_id, "msgContent", reply_md,
                        append=False, finished=True, failed=False)
            c.ai_finish(reply_md)
            log.info("卡片定稿 %d 字", len(reply_md))
        else:
            self.reply_markdown("ragbot", reply_md, msg)
            log.info("整条回复 %d 字", len(reply_md))
        log.info("%s: %s → %d 字 · %dms · %d tokens · rerank%s · rid=%s",
                 nick, question[:40], len(answer), latency_ms, tokens,
                 "正常" if reranked else "降级", rid)


def bot_main():
    """bot 线程入口：建事件循环（非主线程没有现成 loop）→ 连钉钉 → 常驻收消息"""
    cid = os.environ.get("DINGTALK_CLIENT_ID", "")
    secret = os.environ.get("DINGTALK_CLIENT_SECRET", "")
    if not (cid and secret):
        log.info("未配置 DINGTALK_CLIENT_ID/SECRET，钉钉机器人不启动")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = dingtalk_stream.DingTalkStreamClient(dingtalk_stream.Credential(cid, secret))
    client.register_callback_handler(dingtalk_stream.ChatbotMessage.TOPIC, RAGBotHandler())
    log.info("钉钉机器人已启动（Stream 模式，出站连接免公网）· 等待消息…")
    try:
        client.start_forever()          # 阻塞常驻，网络异常自动重连
    except Exception as e:
        log.error("Stream 连接退出：%s", e)
