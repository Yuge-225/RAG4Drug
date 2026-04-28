"""
drug_sql_retriever.py
------------------------------------------
SQLite 精确检索模块：
1) 从用户输入中识别药物名称（LLM + 规则兜底）
2) 将药物名称映射到 DrugBank ID
3) 查询结构化信息（相互作用、食物禁忌、剂量、代谢酶）
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _get_default_paths_and_model() -> tuple[str, str]:
    """优先读取 data_configuration；失败时使用默认值兜底。"""
    default_db = "./drug_structured.db"
    default_model = "gpt-4o-mini"
    try:
        import data_configuration as config  # pylint: disable=import-outside-toplevel

        db_path = getattr(config, "sqlite_db_path", default_db)
        model_name = getattr(config, "sql_ner_model_name", default_model)
        return db_path, model_name
    except Exception as exc:  # pragma: no cover - 环境缺依赖时的兜底分支
        logger.warning("读取 data_configuration 失败，使用默认配置。原因: %s", exc)
        return default_db, default_model


SQLITE_DB_PATH, SQL_NER_MODEL_NAME = _get_default_paths_and_model()

# 常见中文药名到英文检索词的兜底映射（可按需继续扩展）
COMMON_DRUG_ALIASES = {
    "华法林": ["warfarin"],
    "阿司匹林": ["aspirin", "acetylsalicylic acid"],
    "布洛芬": ["ibuprofen"],
    "对乙酰氨基酚": ["acetaminophen", "paracetamol"],
    "二甲双胍": ["metformin"],
    "氯吡格雷": ["clopidogrel"],
    "辛伐他汀": ["simvastatin"],
}


@dataclass
class SchemaInfo:
    drugs_id_col: str
    drugs_name_col: str
    drugs_group_col: str | None
    synonyms_drug_col: str
    synonyms_name_col: str
    interactions_a_col: str
    interactions_b_col: str
    interactions_severity_col: str
    interactions_desc_col: str
    food_drug_col: str
    food_desc_col: str
    dosage_drug_col: str
    dosage_route_col: str | None
    dosage_form_col: str | None
    dosage_strength_col: str | None
    dosage_amount_col: str | None
    enzyme_drug_col: str
    enzyme_name_col: str
    enzyme_action_col: str | None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}


def _pick_col(columns: set[str], candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise ValueError(f"缺少必要字段，候选: {candidates}, 实际: {sorted(columns)}")
    return None


def _resolve_schema(conn: sqlite3.Connection) -> SchemaInfo:
    """兼容当前项目真实 schema 与说明文档里的 schema。"""
    drugs_cols = _table_columns(conn, "drugs")
    syn_cols = _table_columns(conn, "synonyms")
    ix_cols = _table_columns(conn, "drug_interactions")
    food_cols = _table_columns(conn, "food_interactions")
    dose_cols = _table_columns(conn, "dosages")
    enz_cols = _table_columns(conn, "enzymes")

    return SchemaInfo(
        drugs_id_col=_pick_col(drugs_cols, ["id", "drugbank_id"]),
        drugs_name_col=_pick_col(drugs_cols, ["name"]),
        drugs_group_col=_pick_col(drugs_cols, ["drug_group", "groups"], required=False),
        synonyms_drug_col=_pick_col(syn_cols, ["drug_id", "drugbank_id"]),
        synonyms_name_col=_pick_col(syn_cols, ["synonym"]),
        interactions_a_col=_pick_col(ix_cols, ["drug_id_a", "drug_id"]),
        interactions_b_col=_pick_col(ix_cols, ["drug_id_b", "partner_id"]),
        interactions_severity_col=_pick_col(ix_cols, ["severity"]),
        interactions_desc_col=_pick_col(ix_cols, ["description"]),
        food_drug_col=_pick_col(food_cols, ["drug_id", "drugbank_id"]),
        food_desc_col=_pick_col(food_cols, ["description", "interaction"]),
        dosage_drug_col=_pick_col(dose_cols, ["drug_id", "drugbank_id"]),
        dosage_route_col=_pick_col(dose_cols, ["route"], required=False),
        dosage_form_col=_pick_col(dose_cols, ["form"], required=False),
        dosage_strength_col=_pick_col(dose_cols, ["strength"], required=False),
        dosage_amount_col=_pick_col(dose_cols, ["dosage"], required=False),
        enzyme_drug_col=_pick_col(enz_cols, ["drug_id", "drugbank_id"]),
        enzyme_name_col=_pick_col(enz_cols, ["cyp_name", "enzyme_name"]),
        enzyme_action_col=_pick_col(enz_cols, ["action"], required=False),
    )


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    use_path = db_path or SQLITE_DB_PATH
    path = Path(use_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite 文件不存在: {use_path}")
    conn = sqlite3.connect(use_path)
    conn.row_factory = sqlite3.Row
    return conn


def _extract_json_list(text: str) -> list[str]:
    """从 LLM 返回文本中提取 JSON 数组。"""
    if not text:
        return []
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, dict) and isinstance(obj.get("drugs"), list):
            return [str(x).strip() for x in obj["drugs"] if str(x).strip()]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        return []
    try:
        obj = json.loads(match.group(0))
        return [str(x).strip() for x in obj if str(x).strip()] if isinstance(obj, list) else []
    except json.JSONDecodeError:
        return []


def _extract_with_llm(user_input: str, model_name: str = SQL_NER_MODEL_NAME) -> list[str]:
    """
    使用 LLM 提取药物名称。
    若运行环境缺少依赖或调用失败，会抛异常给上层处理。
    """
    from langchain_openai import ChatOpenAI  # pylint: disable=import-outside-toplevel

    llm = ChatOpenAI(model=model_name, temperature=0)
    prompt = (
        "你是药物实体识别助手。"
        "请从用户输入中提取药物名称，返回严格 JSON 数组字符串，不要输出任何解释。\n"
        "要求：\n"
        "1) 仅返回药物名称（中英文都可）\n"
        "2) 去重\n"
        "3) 如果没有药物名，返回 []\n\n"
        f"用户输入：{user_input}"
    )
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return _extract_json_list(content)


def _extract_with_rules(user_input: str) -> list[str]:
    """LLM 失败时的规则兜底，尽量抽取常见药名片段。"""
    text = user_input.strip()
    if not text:
        return []

    # 常见功能词替换为分隔符，便于从中文问句中切出药名
    for marker in ["和", "与", "及", "还有", "可以", "能", "一起", "联用", "同用", "吃", "吗", "？", "?", "。", ",", "，"]:
        text = text.replace(marker, "|")
    candidates = [x.strip() for x in text.split("|") if x.strip()]

    # 英文药名候选（例如 warfarin、acetylsalicylic acid）
    eng = re.findall(r"[A-Za-z][A-Za-z\\-\\s]{1,40}", user_input)
    candidates.extend([x.strip() for x in eng if x.strip()])

    dedup: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        key = c.lower()
        if key not in seen and len(c) >= 2:
            seen.add(key)
            dedup.append(c)
    return dedup


def _is_approved(group_text: str | None) -> bool:
    if not group_text:
        return False
    return "approved" in group_text.lower()


def _expand_alias_candidates(raw_name: str) -> list[str]:
    """给定原始药名，生成一组候选检索词（原词 + 兜底英文别名）。"""
    candidates = [raw_name]
    for alias in COMMON_DRUG_ALIASES.get(raw_name, []):
        if alias not in candidates:
            candidates.append(alias)
    return candidates


def _match_drug_name(conn: sqlite3.Connection, schema: SchemaInfo, raw_name: str) -> dict[str, Any]:
    """
    将一个药物名映射到 DrugBank ID。
    优先级：精确匹配 > 模糊匹配；同分时优先 approved。
    """
    d_id = schema.drugs_id_col
    d_name = schema.drugs_name_col
    d_group = schema.drugs_group_col
    s_drug = schema.synonyms_drug_col
    s_name = schema.synonyms_name_col

    group_select = f", d.{d_group} AS group_text" if d_group else ", '' AS group_text"

    rows: list[sqlite3.Row] = []
    for query_name in _expand_alias_candidates(raw_name):
        # 1) 同义词精确匹配（大小写不敏感）
        sql_exact_syn = f"""
            SELECT DISTINCT d.{d_id} AS drugbank_id, d.{d_name} AS standard_name, s.{s_name} AS hit_name
                   {group_select}
            FROM synonyms s
            JOIN drugs d ON d.{d_id} = s.{s_drug}
            WHERE LOWER(s.{s_name}) = LOWER(?)
            LIMIT 20
        """
        rows = conn.execute(sql_exact_syn, (query_name,)).fetchall()

        # 2) 药物主名精确匹配
        if not rows:
            sql_exact_name = f"""
                SELECT d.{d_id} AS drugbank_id, d.{d_name} AS standard_name, d.{d_name} AS hit_name
                       {group_select}
                FROM drugs d
                WHERE LOWER(d.{d_name}) = LOWER(?)
                LIMIT 20
            """
            rows = conn.execute(sql_exact_name, (query_name,)).fetchall()

        # 3) 模糊匹配（同义词）
        if not rows:
            sql_fuzzy_syn = f"""
                SELECT DISTINCT d.{d_id} AS drugbank_id, d.{d_name} AS standard_name, s.{s_name} AS hit_name
                       {group_select}
                FROM synonyms s
                JOIN drugs d ON d.{d_id} = s.{s_drug}
                WHERE s.{s_name} LIKE ? COLLATE NOCASE
                LIMIT 50
            """
            rows = conn.execute(sql_fuzzy_syn, (f"%{query_name}%",)).fetchall()

        # 4) 模糊匹配（主名）
        if not rows:
            sql_fuzzy_name = f"""
                SELECT d.{d_id} AS drugbank_id, d.{d_name} AS standard_name, d.{d_name} AS hit_name
                       {group_select}
                FROM drugs d
                WHERE d.{d_name} LIKE ? COLLATE NOCASE
                LIMIT 50
            """
            rows = conn.execute(sql_fuzzy_name, (f"%{query_name}%",)).fetchall()

        if rows:
            break

    if not rows:
        return {"name": raw_name, "drugbank_id": None, "standard_name": None}

    # 按是否 approved 优先，其次按名称长度（更接近主药名）
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            0 if _is_approved(r["group_text"]) else 1,
            len(r["standard_name"]) if r["standard_name"] else 999,
        ),
    )
    best = sorted_rows[0]
    return {
        "name": raw_name,
        "drugbank_id": best["drugbank_id"],
        "standard_name": best["standard_name"],
    }


def extract_drug_names(user_input: str, db_path: str | None = None) -> list[dict]:
    """
    从用户输入中提取药物并映射 ID。
    返回格式：
    [{"name": "华法林", "drugbank_id": "DB00682", "standard_name": "Warfarin"}, ...]
    """
    # Step 1: NER 抽取原始药名
    raw_names: list[str] = []
    try:
        raw_names = _extract_with_llm(user_input)
        logger.info("LLM 提取到药物名: %s", raw_names)
    except Exception as exc:  # pragma: no cover - 依赖/网络异常兜底
        logger.warning("LLM NER 失败，使用规则兜底。原因: %s", exc)

    if not raw_names:
        raw_names = _extract_with_rules(user_input)
        logger.info("规则兜底提取药物名: %s", raw_names)

    # 去重
    dedup_names: list[str] = []
    seen: set[str] = set()
    for n in raw_names:
        key = n.strip().lower()
        if key and key not in seen:
            seen.add(key)
            dedup_names.append(n.strip())

    if not dedup_names:
        return []

    # Step 2: 映射到 DrugBank ID
    with _connect(db_path) as conn:
        schema = _resolve_schema(conn)
        mapped = [_match_drug_name(conn, schema, n) for n in dedup_names]

    return mapped


def _query_drug_interactions(conn: sqlite3.Connection, schema: SchemaInfo, drug_ids: list[str]) -> list[dict]:
    if len(drug_ids) < 2:
        return []

    a_col = schema.interactions_a_col
    b_col = schema.interactions_b_col
    sev_col = schema.interactions_severity_col
    desc_col = schema.interactions_desc_col
    d_id = schema.drugs_id_col
    d_name = schema.drugs_name_col

    results: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    sql = f"""
        SELECT di.{a_col} AS drug_a_id,
               di.{b_col} AS drug_b_id,
               da.{d_name} AS drug_a_name,
               db.{d_name} AS drug_b_name,
               di.{sev_col} AS severity,
               di.{desc_col} AS description
        FROM drug_interactions di
        LEFT JOIN drugs da ON da.{d_id} = di.{a_col}
        LEFT JOIN drugs db ON db.{d_id} = di.{b_col}
        WHERE (di.{a_col} = ? AND di.{b_col} = ?)
           OR (di.{a_col} = ? AND di.{b_col} = ?)
    """

    for id_a, id_b in combinations(sorted(set(drug_ids)), 2):
        rows = conn.execute(sql, (id_a, id_b, id_b, id_a)).fetchall()
        for r in rows:
            key = (r["drug_a_id"], r["drug_b_id"], r["description"] or "")
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "drug_a_id": r["drug_a_id"],
                    "drug_a_name": r["drug_a_name"],
                    "drug_b_id": r["drug_b_id"],
                    "drug_b_name": r["drug_b_name"],
                    "severity": (r["severity"] or "unknown"),
                    "description": r["description"] or "",
                }
            )

    return results


def query_drug_interactions(drug_ids: list[str], db_path: str | None = None) -> list[dict]:
    """查询输入药物 ID 的两两相互作用。"""
    if not drug_ids:
        return []
    with _connect(db_path) as conn:
        schema = _resolve_schema(conn)
        return _query_drug_interactions(conn, schema, drug_ids)


def _query_food_interactions(conn: sqlite3.Connection, schema: SchemaInfo, drug_ids: list[str]) -> list[dict]:
    if not drug_ids:
        return []
    f_drug = schema.food_drug_col
    f_desc = schema.food_desc_col
    d_id = schema.drugs_id_col
    d_name = schema.drugs_name_col
    placeholders = ",".join(["?"] * len(drug_ids))
    sql = f"""
        SELECT fi.{f_drug} AS drug_id,
               d.{d_name} AS drug_name,
               fi.{f_desc} AS description
        FROM food_interactions fi
        LEFT JOIN drugs d ON d.{d_id} = fi.{f_drug}
        WHERE fi.{f_drug} IN ({placeholders})
    """
    rows = conn.execute(sql, tuple(drug_ids)).fetchall()
    return [
        {
            "drug_id": r["drug_id"],
            "drug_name": r["drug_name"],
            "description": r["description"] or "",
        }
        for r in rows
    ]


def query_food_interactions(drug_ids: list[str], db_path: str | None = None) -> list[dict]:
    """查询药物食物相互作用。"""
    if not drug_ids:
        return []
    with _connect(db_path) as conn:
        schema = _resolve_schema(conn)
        return _query_food_interactions(conn, schema, drug_ids)


def _query_dosage_info(conn: sqlite3.Connection, schema: SchemaInfo, drug_ids: list[str]) -> list[dict]:
    if not drug_ids:
        return []
    ds_drug = schema.dosage_drug_col
    ds_route = schema.dosage_route_col
    ds_form = schema.dosage_form_col
    ds_strength = schema.dosage_strength_col
    ds_amount = schema.dosage_amount_col
    d_id = schema.drugs_id_col
    d_name = schema.drugs_name_col
    placeholders = ",".join(["?"] * len(drug_ids))

    select_cols = [f"do.{ds_drug} AS drug_id", f"d.{d_name} AS drug_name"]
    if ds_route:
        select_cols.append(f"do.{ds_route} AS route")
    if ds_form:
        select_cols.append(f"do.{ds_form} AS form")
    if ds_strength:
        select_cols.append(f"do.{ds_strength} AS strength")
    if ds_amount:
        select_cols.append(f"do.{ds_amount} AS dosage")

    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM dosages do
        LEFT JOIN drugs d ON d.{d_id} = do.{ds_drug}
        WHERE do.{ds_drug} IN ({placeholders})
        LIMIT 200
    """
    rows = conn.execute(sql, tuple(drug_ids)).fetchall()

    output: list[dict] = []
    for r in rows:
        item = {
            "drug_id": r["drug_id"],
            "drug_name": r["drug_name"],
            "route": r["route"] if "route" in r.keys() else None,
            "form": r["form"] if "form" in r.keys() else None,
            "strength": r["strength"] if "strength" in r.keys() else None,
            "dosage": r["dosage"] if "dosage" in r.keys() else None,
        }
        output.append(item)
    return output


def query_dosage_info(drug_ids: list[str], db_path: str | None = None) -> list[dict]:
    """查询药物剂量信息。"""
    if not drug_ids:
        return []
    with _connect(db_path) as conn:
        schema = _resolve_schema(conn)
        return _query_dosage_info(conn, schema, drug_ids)


def _query_enzyme_info(conn: sqlite3.Connection, schema: SchemaInfo, drug_ids: list[str]) -> list[dict]:
    if not drug_ids:
        return []
    e_drug = schema.enzyme_drug_col
    e_name = schema.enzyme_name_col
    e_action = schema.enzyme_action_col
    d_id = schema.drugs_id_col
    d_name = schema.drugs_name_col
    placeholders = ",".join(["?"] * len(drug_ids))
    action_select = f", e.{e_action} AS action" if e_action else ", '' AS action"
    sql = f"""
        SELECT e.{e_drug} AS drug_id,
               d.{d_name} AS drug_name,
               e.{e_name} AS enzyme_name
               {action_select}
        FROM enzymes e
        LEFT JOIN drugs d ON d.{d_id} = e.{e_drug}
        WHERE e.{e_drug} IN ({placeholders})
        LIMIT 300
    """
    rows = conn.execute(sql, tuple(drug_ids)).fetchall()
    return [
        {
            "drug_id": r["drug_id"],
            "drug_name": r["drug_name"],
            "enzyme_name": r["enzyme_name"],
            "action": r["action"] or None,
        }
        for r in rows
    ]


def query_enzyme_info(drug_ids: list[str], db_path: str | None = None) -> list[dict]:
    """查询药物 CYP450 相关酶信息。"""
    if not drug_ids:
        return []
    with _connect(db_path) as conn:
        schema = _resolve_schema(conn)
        return _query_enzyme_info(conn, schema, drug_ids)


def retrieve_from_sql(user_input: str, db_path: str | None = None) -> dict:
    """
    完整流程：NER -> 映射 ID -> 多表查询
    返回结构：
    {
        "identified_drugs": [...],
        "interactions": [...],
        "food_interactions": [...],
        "dosages": [...],
        "enzymes": [...]
    }
    """
    identified_drugs = extract_drug_names(user_input, db_path=db_path)
    drug_ids = [d["drugbank_id"] for d in identified_drugs if d.get("drugbank_id")]
    unique_ids = sorted(set(drug_ids))

    if not unique_ids:
        return {
            "identified_drugs": identified_drugs,
            "interactions": [],
            "food_interactions": [],
            "dosages": [],
            "enzymes": [],
            "message": "未识别到可映射的药物 ID，请检查药名拼写或改用英文药名。",
        }

    with _connect(db_path) as conn:
        schema = _resolve_schema(conn)
        interactions = _query_drug_interactions(conn, schema, unique_ids)
        food = _query_food_interactions(conn, schema, unique_ids)
        dosages = _query_dosage_info(conn, schema, unique_ids)
        enzymes = _query_enzyme_info(conn, schema, unique_ids)

    message = "ok"
    if len(unique_ids) == 1:
        message = "仅识别到一种药物，返回该药物相关信息。"
    elif len(unique_ids) >= 2 and not interactions:
        message = "已识别药物，但未找到这些药物组合的相互作用记录。"

    return {
        "identified_drugs": identified_drugs,
        "interactions": interactions,
        "food_interactions": food,
        "dosages": dosages,
        "enzymes": enzymes,
        "message": message,
    }


if __name__ == "__main__":
    test_query = "华法林和阿司匹林能一起吃吗？"
    result = retrieve_from_sql(test_query)
    print(json.dumps(result, ensure_ascii=False, indent=2))
