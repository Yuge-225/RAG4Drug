"""
test_rag.py
────────────────────────────────────────────────────────────────
Verify the full RAG pipeline is actually working.
Tests three layers independently:
  1. SQL retrieval  (drug_sql_retriever)
  2. Vector retrieval (ChromaDB)
  3. Full RAG response (retrieve + LLM generate)

Output is written to test_rag.log for inspection.

Run:
  python test_rag.py
"""

import logging
import json
import sys
from pathlib import Path

LOG_PATH = Path("./test_rag.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

TEST_QUERY = "Can warfarin and aspirin be taken together?"

# ══════════════════════════════════════════════════════════════
# Test 1: SQL Retrieval
# ══════════════════════════════════════════════════════════════
def test_sql_retrieval():
    log.info("=" * 60)
    log.info("TEST 1: SQL Retrieval")
    log.info("=" * 60)
    log.info("Query: %s", TEST_QUERY)

    from drug_sql_retriever import retrieve_from_sql
    result = retrieve_from_sql(TEST_QUERY)

    log.info("--- Identified drugs ---")
    for d in result["identified_drugs"]:
        log.info("  name=%-30s  drugbank_id=%s  standard_name=%s",
                 d.get("name"), d.get("drugbank_id"), d.get("standard_name"))

    log.info("--- Interaction records: %d ---", len(result["interactions"]))
    for r in result["interactions"][:5]:
        log.info("  %s x %s  |  severity: %s",
                 r.get("drug_a_name"), r.get("drug_b_name"), r.get("severity"))
        log.info("  description: %s", r.get("description", "")[:150])

    log.info("--- Food interactions: %d ---", len(result["food_interactions"]))
    log.info("--- Message: %s ---", result.get("message"))

    # Pass/Fail
    warfarin_found = any(
        d.get("standard_name", "").lower() == "warfarin"
        for d in result["identified_drugs"]
    )
    aspirin_found = any(
        d.get("standard_name", "").lower() == "acetylsalicylic acid"
        for d in result["identified_drugs"]
    )
    has_interaction = len(result["interactions"]) > 0

    log.info("")
    log.info("RESULT: warfarin_found=%s  aspirin_found=%s  interactions_found=%s",
             warfarin_found, aspirin_found, has_interaction)

    if warfarin_found and aspirin_found and has_interaction:
        log.info("TEST 1 PASSED - SQL retrieval is working correctly")
    else:
        log.warning("TEST 1 FAILED - SQL retrieval has issues")

    return result


# ══════════════════════════════════════════════════════════════
# Test 2: Vector Retrieval (ChromaDB)
# ══════════════════════════════════════════════════════════════
def test_vector_retrieval():
    log.info("")
    log.info("=" * 60)
    log.info("TEST 2: Vector Retrieval (ChromaDB)")
    log.info("=" * 60)
    log.info("Query: %s", TEST_QUERY)

    import data_configuration as config
    from vector_store import VectorStoreService

    vs = VectorStoreService(embedding=config.chosen_embedding_model)
    retriever = vs.get_retriever()
    docs = retriever.invoke(TEST_QUERY)

    log.info("Documents retrieved: %d", len(docs))
    for i, doc in enumerate(docs, 1):
        field     = doc.metadata.get("field", "unknown")
        drug_name = doc.metadata.get("drug_name", "unknown")
        content   = (doc.page_content or "").strip()[:200]
        log.info("  [%d] drug=%s  field=%s", i, drug_name, field)
        log.info("       content: %s", content)

    if docs:
        log.info("TEST 2 PASSED - ChromaDB retrieval is working correctly")
    else:
        log.warning("TEST 2 FAILED - No documents returned from ChromaDB")

    return docs


# ══════════════════════════════════════════════════════════════
# Test 3: Full RAG Response
# ══════════════════════════════════════════════════════════════
def test_full_rag():
    log.info("")
    log.info("=" * 60)
    log.info("TEST 3: Full RAG Response (retrieve + LLM generate)")
    log.info("=" * 60)
    log.info("Query: %s", TEST_QUERY)

    import data_configuration as config
    from rag import RagService

    rag = RagService()

    log.info("--- Retrieved context (before LLM) ---")
    context = rag.retrieve(TEST_QUERY)
    log.info(context)

    log.info("")
    log.info("--- LLM Response ---")
    response = rag.chain.invoke(
        {"question": TEST_QUERY},
        config={"configurable": {"session_id": "test_session"}},
    )
    log.info(response)

    if response and len(response) > 20:
        log.info("")
        log.info("TEST 3 PASSED - Full RAG pipeline is working correctly")
    else:
        log.warning("TEST 3 FAILED - LLM response is empty or too short")

    return response


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("RAG Pipeline Verification")
    log.info("Query: %s", TEST_QUERY)
    log.info("")

    try:
        test_sql_retrieval()
    except Exception as e:
        log.error("TEST 1 ERROR: %s", e, exc_info=True)

    try:
        test_vector_retrieval()
    except Exception as e:
        log.error("TEST 2 ERROR: %s", e, exc_info=True)

    try:
        test_full_rag()
    except Exception as e:
        log.error("TEST 3 ERROR: %s", e, exc_info=True)

    log.info("")
    log.info("Done. Full log saved to: %s", LOG_PATH.resolve())
