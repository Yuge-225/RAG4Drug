"""
data_cleaning.py
────────────────────────────────────────────────────────────────
从 drug_structured.db 清洗后写入两张新表：
  cleaned_drugs          每个药物一行（id, name, description, cyp_enzymes）
  cleaned_interactions   每个唯一药对一行（规范化方向 + 合并描述）

运行方式：
  python data_cleaning.py
"""

import sqlite3
import re
from pathlib import Path

DB_PATH = "./drug_structured.db"


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def strip_html(text: str) -> str:
    """去除 HTML 标签、文献引用编号，规范化空白字符"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[A-Z]\d+(?:,\s*[A-Z]\d+)*\]', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# ══════════════════════════════════════════════════════════════
# Step 1：cleaned_drugs
# ══════════════════════════════════════════════════════════════

def build_cleaned_drugs(conn: sqlite3.Connection) -> int:
    """
    保留字段：id, name, description（去 HTML）, cyp_enzymes（从 enzymes 表聚合）
    其余字段（state, groups, half_life 等）对相互作用预测无语义价值，丢弃。
    """
    cyp_map: dict[str, set] = {}
    for drug_id, cyp_name in conn.execute("SELECT drug_id, cyp_name FROM enzymes"):
        cyp_map.setdefault(drug_id, set()).add(cyp_name)

    conn.execute("DROP TABLE IF EXISTS cleaned_drugs")
    conn.execute("""
        CREATE TABLE cleaned_drugs (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            cyp_enzymes TEXT NOT NULL DEFAULT ''
        )
    """)

    rows = conn.execute("SELECT id, name, description FROM drugs").fetchall()

    to_insert = [
        (
            drug_id,
            name,
            strip_html(description),
            ','.join(sorted(cyp_map.get(drug_id, set())))
        )
        for drug_id, name, description in rows
    ]

    conn.executemany(
        "INSERT INTO cleaned_drugs (id, name, description, cyp_enzymes) VALUES (?, ?, ?, ?)",
        to_insert
    )
    conn.commit()
    return len(to_insert)


# ══════════════════════════════════════════════════════════════
# Step 2：cleaned_interactions
# ══════════════════════════════════════════════════════════════

def build_cleaned_interactions(conn: sqlite3.Connection) -> int:
    """
    规范化药对方向（MIN/MAX 保证 drug_id_a < drug_id_b）
    合并两个方向的描述文本（A→B 与 B→A 拼接）
    severity 冲突时取优先级高的（severe > moderate > minor > unknown）
    """
    conn.execute("DROP TABLE IF EXISTS cleaned_interactions")
    conn.execute("""
        CREATE TABLE cleaned_interactions (
            drug_id_a   TEXT NOT NULL,
            drug_id_b   TEXT NOT NULL,
            severity    TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (drug_id_a, drug_id_b)
        )
    """)

    conn.execute("""
        INSERT INTO cleaned_interactions (drug_id_a, drug_id_b, severity, description)
        SELECT
            MIN(drug_id_a, drug_id_b),
            MAX(drug_id_a, drug_id_b),
            CASE MAX(
                CASE severity
                    WHEN 'severe'   THEN 4
                    WHEN 'moderate' THEN 3
                    WHEN 'minor'    THEN 2
                    ELSE 1
                END
            )
                WHEN 4 THEN 'severe'
                WHEN 3 THEN 'moderate'
                WHEN 2 THEN 'minor'
                ELSE 'unknown'
            END,
            GROUP_CONCAT(description, ' ')
        FROM drug_interactions
        GROUP BY MIN(drug_id_a, drug_id_b), MAX(drug_id_a, drug_id_b)
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_a   ON cleaned_interactions(drug_id_a)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_b   ON cleaned_interactions(drug_id_b)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_sev ON cleaned_interactions(severity)")
    conn.commit()

    return conn.execute("SELECT COUNT(*) FROM cleaned_interactions").fetchone()[0]


# ══════════════════════════════════════════════════════════════
# 结果摘要
# ══════════════════════════════════════════════════════════════

def print_summary(conn: sqlite3.Connection):
    print("\n" + "=" * 52)
    print("清洗结果摘要")
    print("=" * 52)

    total_d = conn.execute("SELECT COUNT(*) FROM cleaned_drugs").fetchone()[0]
    empty_desc = conn.execute(
        "SELECT COUNT(*) FROM cleaned_drugs WHERE description = ''"
    ).fetchone()[0]
    has_cyp = conn.execute(
        "SELECT COUNT(*) FROM cleaned_drugs WHERE cyp_enzymes != ''"
    ).fetchone()[0]

    print(f"\ncleaned_drugs        : {total_d:,} 行")
    print(f"  description 非空   : {total_d - empty_desc:,}  ({(total_d - empty_desc) / total_d:.1%})")
    print(f"  cyp_enzymes 非空   : {has_cyp:,}  ({has_cyp / total_d:.1%})")

    total_i = conn.execute("SELECT COUNT(*) FROM cleaned_interactions").fetchone()[0]
    removed  = 2_910_010 - total_i
    print(f"\ncleaned_interactions : {total_i:,} 行")
    print(f"  原始 2,910,010 → 去重后 {total_i:,}（删除 {removed:,} 条反向重复）")

    rows = conn.execute(
        "SELECT severity, COUNT(*) FROM cleaned_interactions "
        "GROUP BY severity ORDER BY COUNT(*) DESC"
    ).fetchall()
    for sev, cnt in rows:
        print(f"  {sev:10s}: {cnt:>10,}  ({cnt / total_i:.1%})")

    print()
    print("下一步：运行 feature_engineering.py")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if not Path(DB_PATH).exists():
        print(f"找不到数据库: {DB_PATH}")
        exit(1)

    conn = sqlite3.connect(DB_PATH)

    print("Step 1: drugs → cleaned_drugs ...")
    n = build_cleaned_drugs(conn)
    print(f"  完成：{n:,} 行")

    print("\nStep 2: drug_interactions → cleaned_interactions")
    print("  GROUP BY 2.9M 行，约 1-3 分钟...")
    n = build_cleaned_interactions(conn)
    print(f"  完成：{n:,} 行")

    print_summary(conn)
    conn.close()
