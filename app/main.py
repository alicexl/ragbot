"""服务入口：Flask Web 层 + 钉钉 bot 线程

路由（与数据流）：
  GET  /                聊天页（app/static/index.html）
  GET  /api/status      库状态（chunk 数 + 文档清单）
  POST /api/upload      multipart 上传 → 切块/向量化/入库（幂等；PDF/Word 走多模态解析）
  POST /api/clear       清空向量库
  POST /api/chat        SSE 流式回答（sources → delta… → done / error）
  GET  /api/media/<名>   图片 chunk 原图回显

启动：python -X utf8 -m app.main（前提见 README；Milvus 单进程锁，勿开 debug reloader）
"""

import json
import os
import re
import threading
import time
import webbrowser
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from . import limits, llm, pipeline, trace
from .llm import StreamedAnswer
from .store import DATA_DIR, MEDIA_DIR, TOP_K

STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=None)   # 关掉自带 /static 路由，页面统一从 / 进


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/status")
def status():
    """左栏库状态：现有多少 chunk、已导入哪些文档（按来源去重）。"""
    return jsonify(chunks=pipeline.store.chunk_count(),
                   documents=pipeline.store.sources())


@app.post("/api/upload")
def upload():
    """「导入」端点：multipart 真上传。按扩展名分流：.pdf 走多模态解析管线，其余按纯文本。"""
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="未收到文件（multipart 字段名须为 files）"), 400
    notes = []
    for f in files:
        source = Path(f.filename).stem        # source 取文件名（去扩展名，幂等键）
        name_l = f.filename.lower()
        if name_l.endswith(".pdf"):
            notes.append(pipeline.ingest_pdf(f.read(), source)[2])
        elif name_l.endswith(".docx"):
            notes.append(pipeline.ingest_docx(f.read(), source)[2])
        elif name_l.endswith(".doc"):
            notes.append(f"[ERR]「{source}」旧版 .doc 不支持：请在 Word 里转存为 .docx 再上传")
        else:
            notes.append(pipeline.ingest_text(f.read().decode("utf-8", errors="replace"), source)[2])
    return jsonify(chunks=pipeline.store.chunk_count(), notes=notes)


@app.get("/api/media/<name>")
def media(name):
    """图片原件回显：image chunk 的 content 里存文件名，前端经此路由取图（受控目录，无遍历风险）。"""
    return send_from_directory(MEDIA_DIR, name)


@app.post("/api/clear")
def clear():
    """「清空」端点：drop 掉 collection 重建（.db 文件保留，库回到空）。"""
    pipeline.store.clear()
    return jsonify(chunks=0, documents=[])


def _sse(event: str, data: dict) -> str:
    """SSE 帧：event 名 + JSON data 两行，空行结尾（前端按空行切帧）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat():
    """聊天端点。错误分两层：请求级校验（空消息/空库/超限）走 4xx JSON——那时响应头
    还没定；流一旦开始（响应头已是 text/event-stream）中途出错就只能写 SSE error。"""
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    history = body.get("history") or []
    if not message:
        return jsonify(error="message 不能为空"), 400
    if not pipeline.store.chunk_count():
        return jsonify(error="向量库为空，请先在左侧导入文档"), 400
    ok, why = limits.allow("web")
    if not ok:
        return jsonify(error=why), 429
    limits.count_question("web")

    rid = trace.new_request_id()
    t_e2e = time.perf_counter()
    retrieved, reranked = pipeline.retrieve(message, top_k=TOP_K, rid=rid,
                                            history=history)                   # 混合检索
    llm_messages = pipeline.build_user_msg(message, retrieved, history)        # 增强：拼 context
    sources_payload = []
    for i, e in enumerate(retrieved, 1):
        m = re.search(r"原图文件：(.+)$", e.content, re.M) if e.chunk_type == "image" else None
        sources_payload.append({
            "i": i, "rrf": round(e.rrf, 4),
            "score": (round(e.score, 3) if e.score is not None else None),
            "chunk_type": e.chunk_type, "page": e.page, "source": e.source,
            "section": e.section or None,
            "preview": e.text.replace("\n", " ").strip()[:80],
            "image": m.group(1) if m else None,      # 图片 chunk → 文件名，前端回显原图
        })

    def stream():
        yield _sse("sources", {"sources": sources_payload, "reranked": reranked, "rid": rid})
        try:
            sa = StreamedAnswer(llm.SYSTEM_PROMPT, llm_messages, rid=rid,
                                thinking_off=True)   # 关思考：与钉钉一致——首字快 + 省 token（评测同口径）
            for delta in sa:
                yield _sse("delta", {"text": delta})
            limits.count_tokens(sa.usage)
            trace.record(rid, "e2e", (time.perf_counter() - t_e2e) * 1000,
                         meta={"tokens": sa.usage})
            yield _sse("done", {})
        except Exception as e:
            yield _sse("error", {"message": f"生成失败：{e}\n请稍后重试或联系管理员"})

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache"})


def main():
    pipeline.init()

    print("=" * 60)
    print(f"ragbot · 检索 embedding-3 · 生成 {pipeline.generator.model}")
    print(f"向量库：{pipeline.store.chunk_count()} 个 chunk · Milvus Lite（data/mini_rag.db）")
    print("浏览器打开 http://127.0.0.1:7860 · Ctrl+C 退出")
    if os.environ.get("DINGTALK_CLIENT_ID") and os.environ.get("DINGTALK_CLIENT_SECRET"):
        print("检测到钉钉凭据：机器人线程将随服务启动")
    print("=" * 60)
    # 钉钉机器人（可选）：同进程线程（Milvus 单进程锁，bot 与 Web 必须共库同进程）
    if os.environ.get("DINGTALK_CLIENT_ID") and os.environ.get("DINGTALK_CLIENT_SECRET"):
        from . import bot
        threading.Thread(target=bot.bot_main, daemon=True, name="dingtalk-bot").start()

    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    app.run(host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
