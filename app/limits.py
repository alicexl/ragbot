"""限流与预算：按人每日问数 + 全局每日 token 预算（防单人烧穿配额）

计数落 SQLite（短连接），按自然日重置（day 作主键的一部分，过期行懒得删也不影响正确性）。
超限不给「错误」，给「友好解释」——限流是产品行为不是故障。
env 可调：DAILY_QUESTION_LIMIT（默认 50）/ DAILY_TOKEN_BUDGET（默认 500k tokens）。
"""

import os
import sqlite3
import time
from pathlib import Path

LIMITS_DB = str(Path(__file__).parent.parent / "data" / "limits.db")

DAILY_QUESTION_LIMIT = int(os.environ.get("DAILY_QUESTION_LIMIT", 50))
DAILY_TOKEN_BUDGET = int(os.environ.get("DAILY_TOKEN_BUDGET", 500_000))


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(LIMITS_DB)
    c.execute("""CREATE TABLE IF NOT EXISTS usage(
        day TEXT NOT NULL,
        user TEXT NOT NULL,
        questions INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (day, user))""")
    c.execute("""CREATE TABLE IF NOT EXISTS tokens(
        day TEXT NOT NULL PRIMARY KEY,
        total INTEGER NOT NULL DEFAULT 0)""")
    return c


def _day() -> str:
    return time.strftime("%Y-%m-%d")


def allow(user: str) -> tuple[bool, str]:
    """放行判断：该用户今日问数 + 全局今日 token 预算。放行后记得回填 count_*。"""
    with _conn() as c:
        q = c.execute("SELECT questions FROM usage WHERE day=? AND user=?",
                      (_day(), user)).fetchone()
        used_q = q[0] if q else 0
        t = c.execute("SELECT total FROM tokens WHERE day=?", (_day(),)).fetchone()
        used_t = t[0] if t else 0
    if used_q >= DAILY_QUESTION_LIMIT:
        return False, f"今日提问额度已用完（{DAILY_QUESTION_LIMIT} 问/人/天），明天再来吧。"
    if used_t >= DAILY_TOKEN_BUDGET:
        return False, "今日全局 token 预算已耗尽（服务配额保护），明天再来吧。"
    return True, ""


def count_question(user: str) -> None:
    with _conn() as c:
        c.execute("""INSERT INTO usage(day, user, questions) VALUES(?,?,1)
                     ON CONFLICT(day, user) DO UPDATE SET questions = questions + 1""",
                  (_day(), user))


def count_tokens(n: int) -> None:
    with _conn() as c:
        c.execute("""INSERT INTO tokens(day, total) VALUES(?,?)
                     ON CONFLICT(day) DO UPDATE SET total = total + ?""",
                  (_day(), n, n))


def today_usage() -> dict:
    """今日用量（状态展示 / 巡检用）"""
    with _conn() as c:
        users = c.execute("SELECT user, questions FROM usage WHERE day=?", (_day(),)).fetchall()
        t = c.execute("SELECT total FROM tokens WHERE day=?", (_day(),)).fetchone()
    return {"questions": dict(users), "tokens": t[0] if t else 0,
            "question_limit": DAILY_QUESTION_LIMIT, "token_budget": DAILY_TOKEN_BUDGET}
