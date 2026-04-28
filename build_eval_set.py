"""
build_eval_set.py
────────────────────────────────────────────────────────────────
Samples drug pairs from drug_structured.db and generates
a JSON evaluation set for the RAG vs LLM-alone comparison.

Ground truth source: DrugBank (via cleaned_interactions + drugs tables)

Categories (50 total):
  A. severe      6  pairs  — Azathioprine + ACE inhibitors (all 27 are this class)
  B. moderate   20  pairs  — random diverse sample
  C. minor       5  pairs  — random sample
  D. no_record  10  pairs  — real approved drugs with NO interaction record in DrugBank
  E. multi_drug  9  triples — 3 drug sets × 3 question variants

Output: eval_set.json

Run:
  python build_eval_set.py
"""

import sqlite3
import json
import random
from pathlib import Path
from itertools import combinations

DB_PATH    = "./drug_structured.db"
OUTPUT     = Path("./eval_set.json")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════
# Question templates
# ══════════════════════════════════════════════════════════════

PAIR_TEMPLATES = [
    "Can {a} and {b} be taken together?",
    "Is it safe to combine {a} and {b}?",
    "What are the drug interactions between {a} and {b}?",
    "Do {a} and {b} interact with each other?",
    "I am currently taking {a}. My doctor also prescribed {b}. Are there any concerns?",
    "What should I know about taking {a} and {b} at the same time?",
    "Are there any risks in using {a} together with {b}?",
    "I need to take both {a} and {b}. Is this combination safe?",
]

MULTI_TEMPLATES = [
    "I am taking {a}, {b}, and {c}. Are there any drug interactions I should know about?",
    "What are the potential interactions between {a}, {b}, and {c}?",
    "Is it safe to take {a}, {b}, and {c} together?",
]


def pick_template(templates: list, used: set) -> str:
    """Pick a template not recently used to maximise variety."""
    available = [t for t in templates if t not in used]
    if not available:
        used.clear()
        available = templates
    chosen = random.choice(available)
    used.add(chosen)
    return chosen


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def fetch_pairs(conn: sqlite3.Connection, severity: str, limit: int) -> list:
    return conn.execute("""
        SELECT ci.drug_id_a, ci.drug_id_b,
               ci.severity, ci.description,
               da.name AS name_a, db.name AS name_b
        FROM cleaned_interactions ci
        JOIN drugs da ON da.id = ci.drug_id_a
        JOIN drugs db ON db.id = ci.drug_id_b
        WHERE ci.severity = ?
        ORDER BY RANDOM()
        LIMIT ?
    """, (severity, limit)).fetchall()


def has_interaction(conn: sqlite3.Connection, id_a: str, id_b: str) -> dict | None:
    """Return interaction record if it exists (either direction), else None."""
    row = conn.execute("""
        SELECT severity, description
        FROM cleaned_interactions
        WHERE (drug_id_a = ? AND drug_id_b = ?)
           OR (drug_id_a = ? AND drug_id_b = ?)
        LIMIT 1
    """, (id_a, id_b, id_b, id_a)).fetchone()
    if row:
        return {"severity": row[0], "description": row[1]}
    return None


def make_pair_item(idx: int, category: str, row, template: str) -> dict:
    """Build one eval set entry for a two-drug pair."""
    name_a, name_b = row[4], row[5]
    return {
        "id":       f"eval_{idx:03d}",
        "category": category,
        "question": template.format(a=name_a, b=name_b),
        "drug_a":   name_a,
        "drug_b":   name_b,
        "drug_a_id": row[0],
        "drug_b_id": row[1],
        "ground_truth": {
            "has_record":  True,
            "severity":    row[2],
            "description": row[3],
        },
    }


# ══════════════════════════════════════════════════════════════
# Category samplers
# ══════════════════════════════════════════════════════════════

def sample_known(conn, severity: str, n: int, start_idx: int,
                 category: str, used_tpl: set) -> list:
    rows = fetch_pairs(conn, severity, n)
    items = []
    for i, row in enumerate(rows):
        tpl = pick_template(PAIR_TEMPLATES, used_tpl)
        items.append(make_pair_item(start_idx + i, category, row, tpl))
    return items


def sample_no_record(conn, n: int, start_idx: int, used_tpl: set) -> list:
    """
    Find drug pairs where NEITHER direction exists in cleaned_interactions.
    Source: approved drugs with non-empty descriptions.
    """
    approved = conn.execute("""
        SELECT id, name FROM drugs
        WHERE groups LIKE '%approved%'
          AND description IS NOT NULL AND description != ''
        ORDER BY RANDOM()
        LIMIT 500
    """).fetchall()

    items = []
    seen_pairs: set = set()

    for d_a, d_b in combinations(approved, 2):
        if len(items) >= n:
            break
        key = (min(d_a[0], d_b[0]), max(d_a[0], d_b[0]))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        record = has_interaction(conn, d_a[0], d_b[0])
        if record is None:
            tpl = pick_template(PAIR_TEMPLATES, used_tpl)
            items.append({
                "id":       f"eval_{start_idx + len(items):03d}",
                "category": "no_record",
                "question": tpl.format(a=d_a[1], b=d_b[1]),
                "drug_a":   d_a[1],
                "drug_b":   d_b[1],
                "drug_a_id": d_a[0],
                "drug_b_id": d_b[0],
                "ground_truth": {
                    "has_record":  False,
                    "severity":    None,
                    "description": None,
                    "note": "No interaction record found in DrugBank. "
                            "System should acknowledge absence of data rather than fabricate.",
                },
            })

    print(f"  no_record: found {len(items)} pairs")
    return items


def sample_multi_drug(conn, n_triples: int, start_idx: int) -> list:
    """
    Pick n_triples sets of 3 drugs from moderate interactions.
    For each triple, look up all 3 pairwise ground truths.
    Each triple generates 3 question variants (one per MULTI_TEMPLATE).
    """
    # Get a pool of diverse drugs involved in moderate interactions
    pool_rows = conn.execute("""
        SELECT DISTINCT ci.drug_id_a, da.name
        FROM cleaned_interactions ci
        JOIN drugs da ON da.id = ci.drug_id_a
        WHERE ci.severity = 'moderate'
        ORDER BY RANDOM()
        LIMIT 60
    """).fetchall()

    pool = list(pool_rows)
    random.shuffle(pool)

    items = []
    used_drug_sets: list = []

    for triple in combinations(pool, 3):
        if len(used_drug_sets) >= n_triples:
            break

        ids   = [t[0] for t in triple]
        names = [t[1] for t in triple]

        # Ensure at least one pair has a known interaction
        pair_gts = []
        has_any = False
        for (ia, na), (ib, nb) in combinations(zip(ids, names), 2):
            record = has_interaction(conn, ia, ib)
            if record:
                has_any = True
                pair_gts.append({
                    "drug_a":      na,
                    "drug_b":      nb,
                    "has_record":  True,
                    "severity":    record["severity"],
                    "description": record["description"],
                })
            else:
                pair_gts.append({
                    "drug_a":      na,
                    "drug_b":      nb,
                    "has_record":  False,
                    "severity":    None,
                    "description": None,
                })

        if not has_any:
            continue

        used_drug_sets.append(ids)

        # One entry per MULTI_TEMPLATE
        for tpl in MULTI_TEMPLATES:
            items.append({
                "id":       f"eval_{start_idx + len(items):03d}",
                "category": "multi_drug",
                "question": tpl.format(a=names[0], b=names[1], c=names[2]),
                "drugs":    names,
                "drug_ids": ids,
                "ground_truth": {
                    "pairs": pair_gts,
                },
            })

        if len(items) >= n_triples * len(MULTI_TEMPLATES):
            break

    print(f"  multi_drug: {len(used_drug_sets)} triples × "
          f"{len(MULTI_TEMPLATES)} templates = {len(items)} items")
    return items


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    eval_set = []
    used_tpl: set = set()   # track recently used templates for variety

    print("Building evaluation set...")

    # A. Severe (6)
    print("  [A] severe...")
    items = sample_known(conn, "severe", 6, len(eval_set), "severe", used_tpl)
    eval_set.extend(items)
    print(f"      → {len(items)} items")

    # B. Moderate (20)
    print("  [B] moderate...")
    items = sample_known(conn, "moderate", 20, len(eval_set), "moderate", used_tpl)
    eval_set.extend(items)
    print(f"      → {len(items)} items")

    # C. Minor (5)
    print("  [C] minor...")
    items = sample_known(conn, "minor", 5, len(eval_set), "minor", used_tpl)
    eval_set.extend(items)
    print(f"      → {len(items)} items")

    # D. No Record (10)
    print("  [D] no_record...")
    items = sample_no_record(conn, 10, len(eval_set), used_tpl)
    eval_set.extend(items)

    # E. Multi-drug (3 triples × 3 templates = 9)
    print("  [E] multi_drug...")
    items = sample_multi_drug(conn, n_triples=3, start_idx=len(eval_set))
    eval_set.extend(items)

    conn.close()

    # Write output
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*50}")
    print(f"Total items: {len(eval_set)}")
    from collections import Counter
    cats = Counter(item["category"] for item in eval_set)
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat:12s}: {cnt}")
    print(f"\nSaved to: {OUTPUT}")

    # Preview first item of each category
    print(f"\n{'='*50}")
    print("Preview (first item per category):")
    seen_cats: set = set()
    for item in eval_set:
        cat = item["category"]
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        print(f"\n[{cat}]")
        print(f"  Q : {item['question']}")
        if cat == "multi_drug":
            for p in item["ground_truth"]["pairs"]:
                rec = "✓" if p["has_record"] else "✗"
                print(f"  GT: {rec} {p['drug_a']} × {p['drug_b']} → {p['severity']}")
        else:
            gt = item["ground_truth"]
            print(f"  GT: has_record={gt['has_record']}  severity={gt['severity']}")
            if gt["description"]:
                print(f"      {gt['description'][:100]}")


if __name__ == "__main__":
    main()
