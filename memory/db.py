"""
memory/db.py
SQLite storage for VISION's conversation history and durable facts.
Conversation history: recent messages, expires from active context over time.
Facts: durable info (name, preferences) that's always loaded, regardless of age.
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config


def get_connection():
    config.MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            task_type TEXT,
            model_used TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("[memory] Database ready.")


def save_message(role: str, content: str, task_type: str = None, model_used: str = None):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO messages (timestamp, role, content, task_type, model_used)
        VALUES (?, ?, ?, ?, ?)
        """,
        (datetime.now().isoformat(), role, content, task_type, model_used),
    )
    conn.commit()
    conn.close()


def get_recent_history(limit: int = 10) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def clear_history():
    conn = get_connection()
    conn.execute("DELETE FROM messages")
    conn.commit()
    conn.close()
    print("[memory] History cleared.")


# ── Durable facts ──
def save_fact(key: str, value: str):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO facts (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key.strip().lower(), value.strip(), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_all_facts() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
    conn.close()
    return [{"key": row["key"], "value": row["value"]} for row in rows]


def delete_fact(key: str):
    conn = get_connection()
    conn.execute("DELETE FROM facts WHERE key = ?", (key.strip().lower(),))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    save_fact("name", "David")
    save_fact("dietary_preference", "vegetarian")
    print("\nFacts:")
    for f in get_all_facts():
        print(f"  {f['key']}: {f['value']}")