"""
evaluate.py
────────────────────────────────────────────────────────────────
LLM Alone vs RAG evaluation on eval_set.json.

For each question:
  1. LLM-alone response  (no context, raw GPT-4o-mini)
  2. RAG response        (SQL + vector retrieval + LLM)
  3. LLM-as-Judge scores both against DrugBank ground truth

Metrics (each 0–3 or 0/1):
  hallucination_score : fabrication level    (3=none, 0=severe)
  faithfulness_score  : alignment with GT    (3=full, 0=contradicts)
  completeness_score  : coverage of details  (3=comprehensive)
  severity_accuracy   : correct severity     (1/0, N/A for no_record)
  no_record_refusal   : correctly declined   (1/0, N/A for has_record)

Output:
  eval_results.json   raw scores + responses (checkpoint)
  eval_results.xlsx   3-sheet Excel report

Run:
  python evaluate.py
"""

import json
import time
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

EVAL_SET_PATH = Path("./eval_set.json")
RESULTS_PATH  = Path("./eval_results.json")
EXCEL_PATH    = Path("./eval_results.xlsx")
SLEEP_SEC     = 1.2   # seconds between API calls


# ══════════════════════════════════════════════════════════════
# LLM Alone (no retrieval context)
# ══════════════════════════════════════════════════════════════

LLM_ALONE_SYSTEM = (
    "You are a clinical pharmacist AI assistant. "
    "Answer questions about drug interactions based on your training knowledge. "
    "Be specific about severity levels (Major/Moderate/Minor) when you know them. "
    "If you do not have reliable information about a specific drug combination, "
    "say so clearly rather than guessing."
)

def get_llm_alone_response(llm: ChatOpenAI, question: str) -> str:
    messages = [
        SystemMessage(content=LLM_ALONE_SYSTEM),
        HumanMessage(content=question),
    ]
    return llm.invoke(messages).content


# ══════════════════════════════════════════════════════════════
# RAG Response
# ══════════════════════════════════════════════════════════════

def get_rag_response(rag_service, question: str, session_id: str) -> tuple:
    """Returns (response_text, context_str)."""
    context = rag_service.retrieve(question)
    response = rag_service.chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )
    return response, context


# ══════════════════════════════════════════════════════════════
# LLM-as-Judge
# ══════════════════════════════════════════════════════════════

JUDGE_SYSTEM = (
    "You are an expert clinical pharmacology judge. "
    "Evaluate AI drug-interaction responses against DrugBank ground truth. "
    "Return ONLY valid JSON — no markdown, no extra text."
)

JUDGE_TEMPLATE = """\
QUESTION: {question}
CATEGORY: {category}

DRUGBANK GROUND TRUTH:
{gt_json}

AI RESPONSE TO EVALUATE:
{response}

Score this response. Return a JSON object with exactly these keys:

{{
  "hallucination_score": <integer 0-3>,
  "faithfulness_score":  <integer 0-3>,
  "completeness_score":  <integer 0-3>,
  "severity_accuracy":   <1 or 0 or null>,
  "no_record_refusal":   <1 or 0 or null>,
  "reasoning":           "<1-2 sentence justification>"
}}

Scoring rules:
hallucination_score:
  3 = no fabricated claims (everything stated is supported by GT or general pharmacology)
  2 = one or two minor unsupported details (e.g. mechanism not in GT but plausible)
  1 = notable unsupported claims that could mislead a clinician
  0 = significant fabrication (e.g. inventing an interaction that does not exist)

faithfulness_score:
  3 = response is fully consistent with DrugBank description
  2 = mostly consistent; minor omissions or slight paraphrasing
  1 = partially consistent; key details missing or altered
  0 = contradicts or completely ignores ground truth

completeness_score:
  3 = covers severity, mechanism, clinical advice
  2 = covers most key points; one area thin
  1 = covers some aspects; significant gaps
  0 = very superficial or off-topic

severity_accuracy:
  Set to 1 if the response correctly identifies the severity level matching GT (severe/moderate/minor).
  Set to 0 if severity is wrong or not mentioned when GT has a record.
  Set to null if category is "no_record".

no_record_refusal:
  Set to 1 if category is "no_record" AND response clearly states no interaction data found
    (phrases like "no record", "no data", "not found in database", "cannot confirm" count).
  Set to 0 if category is "no_record" AND response fabricates an interaction.
  Set to null for all other categories.

For multi_drug: evaluate against ALL pairs in the ground truth pairs list.\
"""

def _strip_markdown(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def judge_response(judge_llm: ChatOpenAI, question: str, ground_truth: dict,
                   category: str, response: str) -> dict:
    gt_json = json.dumps(ground_truth, ensure_ascii=False, indent=2)
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        category=category,
        gt_json=gt_json,
        response=response,
    )
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=prompt),
    ]
    raw = judge_llm.invoke(messages).content
    raw = _strip_markdown(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "hallucination_score": None,
            "faithfulness_score":  None,
            "completeness_score":  None,
            "severity_accuracy":   None,
            "no_record_refusal":   None,
            "reasoning": f"[JSON parse error] {raw[:300]}",
            "parse_error": True,
        }


# ══════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════

def run_evaluation() -> list:
    from rag import RagService

    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(eval_set)} evaluation items.\n")

    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} items already done.\n")
    else:
        results  = []
        done_ids = set()

    llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("Initialising RAG service...")
    rag = RagService()
    print("RAG ready.\n")

    total = len(eval_set)
    for i, item in enumerate(eval_set, 1):
        item_id  = item["id"]
        question = item["question"]
        category = item["category"]
        gt       = item["ground_truth"]

        if item_id in done_ids:
            print(f"[{i:02d}/{total}] {item_id} — skip")
            continue

        print(f"[{i:02d}/{total}] {item_id} [{category:10s}] {question[:65]}")

        # 1. LLM alone
        try:
            llm_resp = get_llm_alone_response(llm, question)
        except Exception as e:
            llm_resp = f"ERROR: {e}"
        time.sleep(SLEEP_SEC)

        # 2. RAG
        try:
            rag_resp, rag_ctx = get_rag_response(rag, question, session_id=item_id)
        except Exception as e:
            rag_resp = f"ERROR: {e}"
            rag_ctx  = ""
        time.sleep(SLEEP_SEC)

        # 3. Judge LLM alone
        try:
            llm_scores = judge_response(judge_llm, question, gt, category, llm_resp)
        except Exception as e:
            llm_scores = {"reasoning": f"judge error: {e}"}
        time.sleep(SLEEP_SEC)

        # 4. Judge RAG
        try:
            rag_scores = judge_response(judge_llm, question, gt, category, rag_resp)
        except Exception as e:
            rag_scores = {"reasoning": f"judge error: {e}"}
        time.sleep(SLEEP_SEC)

        record = {
            "id":           item_id,
            "category":     category,
            "question":     question,
            "ground_truth": gt,
            "llm_response": llm_resp,
            "rag_response": rag_resp,
            "rag_context":  rag_ctx[:3000],
            "llm_scores":   llm_scores,
            "rag_scores":   rag_scores,
        }
        results.append(record)
        RESULTS_PATH.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        lh = llm_scores.get("hallucination_score")
        lf = llm_scores.get("faithfulness_score")
        rh = rag_scores.get("hallucination_score")
        rf = rag_scores.get("faithfulness_score")
        print(f"         LLM hall={lh} faith={lf}  |  RAG hall={rh} faith={rf}")

    print(f"\nAll done. {len(results)} results in {RESULTS_PATH}")
    return results


# ══════════════════════════════════════════════════════════════
# Aggregation helpers
# ══════════════════════════════════════════════════════════════

def safe_mean(values: list) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def collect_metric(results: list, system: str, key: str,
                   category: str | None = None) -> list:
    subset = results if category is None else [r for r in results if r["category"] == category]
    return [r.get(system, {}).get(key) for r in subset]


# ══════════════════════════════════════════════════════════════
# Excel report
# ══════════════════════════════════════════════════════════════

def build_excel(results: list):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, numbers
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()

    HDR_FILL  = PatternFill("solid", fgColor="1F4E79")
    LLM_FILL  = PatternFill("solid", fgColor="2E75B6")
    RAG_FILL  = PatternFill("solid", fgColor="70AD47")
    WHITE_FT  = Font(color="FFFFFF", bold=True)
    CENTER    = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def style_header_row(ws, row_num, fill):
        for cell in ws[row_num]:
            cell.fill  = fill
            cell.font  = WHITE_FT
            cell.alignment = CENTER

    CATEGORIES = ["severe", "moderate", "minor", "no_record", "multi_drug", "ALL"]
    METRICS = [
        # (display_name,        key,                 multiplier)
        ("Severity Accuracy (%)",     "severity_accuracy",   100),
        ("Hallucination Score (0-3)", "hallucination_score",   1),
        ("Faithfulness Score (0-3)",  "faithfulness_score",    1),
        ("Completeness Score (0-3)",  "completeness_score",    1),
        ("No-Record Refusal (%)",     "no_record_refusal",   100),
    ]

    # ── Sheet 1: Summary ──────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Summary"
    ws1.freeze_panes = "B3"

    row1 = [""]
    row2 = ["Metric"]
    for cat in CATEGORIES:
        row1 += [cat, ""]
        row2 += ["LLM", "RAG"]
    ws1.append(row1)
    ws1.append(row2)

    # merge category header cells
    col = 2
    for cat in CATEGORIES:
        ltr = get_column_letter(col)
        ws1.merge_cells(f"{ltr}1:{get_column_letter(col+1)}1")
        col += 2

    for mname, mkey, mult in METRICS:
        row = [mname]
        for cat in CATEGORIES:
            subset = results if cat == "ALL" else [r for r in results if r["category"] == cat]
            for sys_key in ("llm_scores", "rag_scores"):
                vals = [r.get(sys_key, {}).get(mkey) for r in subset]
                m = safe_mean(vals)
                row.append(round(m * mult, 2) if m is not None else "N/A")
        ws1.append(row)

    style_header_row(ws1, 1, HDR_FILL)
    # alternating LLM/RAG colours in row 2
    for j, cell in enumerate(ws1[2]):
        if j == 0:
            cell.fill = HDR_FILL
        elif j % 2 == 1:
            cell.fill = LLM_FILL
        else:
            cell.fill = RAG_FILL
        cell.font = WHITE_FT
        cell.alignment = CENTER

    ws1.column_dimensions["A"].width = 28
    for col_idx in range(2, 2 + len(CATEGORIES) * 2):
        ws1.column_dimensions[get_column_letter(col_idx)].width = 10

    # Sample counts block
    from collections import Counter
    ws1.append([])
    ws1.append(["Category", "N items"])
    cats_cnt = Counter(r["category"] for r in results)
    for cat in CATEGORIES[:-1]:
        ws1.append([cat, cats_cnt.get(cat, 0)])
    ws1.append(["TOTAL", len(results)])

    # ── Sheet 2: Per-Question Results ─────────────────────────
    ws2 = wb.create_sheet("Per-Question Results")
    ws2.freeze_panes = "A2"

    hdrs = [
        "ID", "Category", "Question",
        "GT Severity", "GT Has Record",
        "LLM Hallucination", "LLM Faithfulness", "LLM Completeness",
        "LLM Sev Acc", "LLM No-Rec Refusal", "LLM Reasoning",
        "RAG Hallucination", "RAG Faithfulness", "RAG Completeness",
        "RAG Sev Acc", "RAG No-Rec Refusal", "RAG Reasoning",
    ]
    ws2.append(hdrs)
    style_header_row(ws2, 1, HDR_FILL)

    for r in results:
        gt = r["ground_truth"]
        if "pairs" in gt:
            sev_str = " | ".join(
                f"{p['drug_a']}×{p['drug_b']}:{p['severity']}"
                for p in gt["pairs"]
            )
            has_rec = any(p["has_record"] for p in gt["pairs"])
        else:
            sev_str = gt.get("severity")
            has_rec = gt.get("has_record")

        ls = r.get("llm_scores", {})
        rs = r.get("rag_scores", {})

        ws2.append([
            r["id"], r["category"], r["question"],
            sev_str, has_rec,
            ls.get("hallucination_score"), ls.get("faithfulness_score"),
            ls.get("completeness_score"),  ls.get("severity_accuracy"),
            ls.get("no_record_refusal"),   ls.get("reasoning", ""),
            rs.get("hallucination_score"), rs.get("faithfulness_score"),
            rs.get("completeness_score"),  rs.get("severity_accuracy"),
            rs.get("no_record_refusal"),   rs.get("reasoning", ""),
        ])

    widths2 = {"A": 10, "B": 12, "C": 40, "D": 22, "E": 12,
               "F": 14, "G": 14, "H": 14, "I": 12, "J": 15, "K": 38,
               "L": 14, "M": 14, "N": 14, "O": 12, "P": 15, "Q": 38}
    for col, w in widths2.items():
        ws2.column_dimensions[col].width = w

    # ── Sheet 3: Raw Responses ────────────────────────────────
    ws3 = wb.create_sheet("Raw Responses")
    ws3.freeze_panes = "A2"

    ws3.append(["ID", "Category", "Question", "LLM Response", "RAG Response"])
    style_header_row(ws3, 1, HDR_FILL)

    for r in results:
        ws3.append([
            r["id"], r["category"], r["question"],
            r.get("llm_response", ""),
            r.get("rag_response", ""),
        ])
    for col, w in [("A", 10), ("B", 12), ("C", 40), ("D", 65), ("E", 65)]:
        ws3.column_dimensions[col].width = w

    wb.save(EXCEL_PATH)
    print(f"Excel report saved → {EXCEL_PATH}")


# ══════════════════════════════════════════════════════════════
# Terminal summary
# ══════════════════════════════════════════════════════════════

def print_summary(results: list):
    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY  (higher = better for all metrics)")
    print("=" * 55)
    metric_labels = [
        ("Hallucination Score (0-3)", "hallucination_score"),
        ("Faithfulness Score  (0-3)", "faithfulness_score"),
        ("Completeness Score  (0-3)", "completeness_score"),
    ]
    print(f"{'Metric':<28} {'LLM':>8} {'RAG':>8}  {'Delta':>8}")
    print("-" * 55)
    for label, key in metric_labels:
        llm_m = safe_mean(collect_metric(results, "llm_scores", key))
        rag_m = safe_mean(collect_metric(results, "rag_scores", key))
        if llm_m is not None and rag_m is not None:
            delta = rag_m - llm_m
            print(f"{label:<28} {llm_m:>8.4f} {rag_m:>8.4f}  {delta:>+8.4f}")

    # Severity accuracy (has_record items only)
    has_rec = [r for r in results if r["ground_truth"].get("has_record") is True]
    if has_rec:
        llm_sa = safe_mean([r["llm_scores"].get("severity_accuracy") for r in has_rec])
        rag_sa = safe_mean([r["rag_scores"].get("severity_accuracy") for r in has_rec])
        if llm_sa is not None and rag_sa is not None:
            print(f"{'Severity Accuracy (%)':<28} {llm_sa*100:>7.1f}% {rag_sa*100:>7.1f}%  {(rag_sa-llm_sa)*100:>+7.1f}%")

    # No-record refusal (no_record items only)
    no_rec = [r for r in results if r["category"] == "no_record"]
    if no_rec:
        llm_nr = safe_mean([r["llm_scores"].get("no_record_refusal") for r in no_rec])
        rag_nr = safe_mean([r["rag_scores"].get("no_record_refusal") for r in no_rec])
        if llm_nr is not None and rag_nr is not None:
            print(f"{'No-Record Refusal (%)':<28} {llm_nr*100:>7.1f}% {rag_nr*100:>7.1f}%  {(rag_nr-llm_nr)*100:>+7.1f}%")

    print("=" * 55)


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(f"RAG Evaluation  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    results = run_evaluation()

    print("\nBuilding Excel report...")
    build_excel(results)

    print_summary(results)
    print("\nFiles written:")
    print(f"  {RESULTS_PATH}")
    print(f"  {EXCEL_PATH}")
