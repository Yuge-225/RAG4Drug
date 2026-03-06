"""
药物相互作用检索 RAG Pipeline 工业级评估脚本
Drug Interaction Retrieval - Advanced RAG Evaluation

业务场景：
  临床医生/药师查询两种或多种药物之间的相互作用，系统从药典、
  FDA 说明书、PubMed 文献等知识库中检索相关信息并生成结构化回答。

包含：
  1. 药物相互作用黄金数据集自动构建 (Synthetic Golden Dataset)
  2. 检索指标计算 (Hit Rate @ K, MRR)
  3. 生成质量评分 (Faithfulness, Clinical Correctness via LLM-as-a-Judge)
  4. A/B 测试对比 (Top-K / 检索策略)
  5. 安全性专项评估 (Severity Grading Accuracy - HIGH/MODERATE/LOW)
"""

import json
import time
import random
import os
from datetime import datetime
import pandas as pd

# 项目内部模块
from vector_store import VectorStoreService
from rag import RagService
import data_configuration as config
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


# ============================================================
# 【业务配置】药物相互作用场景参数
# ============================================================
DRUG_EVAL_CONFIG = {
    # 测试用药物对列表（可根据实际知识库内容替换）
    "drug_pairs": [
        ("华法林", "阿司匹林"),
        ("辛伐他汀", "胺碘酮"),
        ("地高辛", "奎尼丁"),
        ("氟西汀", "曲马多"),
        ("甲氨蝶呤", "布洛芬"),
        ("卡马西平", "口服避孕药"),
        ("利福平", "酮康唑"),
        ("西地那非", "硝酸甘油"),
        ("锂盐", "氢氯噻嗪"),
        ("克拉霉素", "特非那定"),
    ],
    # 严重程度标签（用于安全性专项评估）
    "severity_labels": ["禁忌 (Contraindicated)", "严重 (Major)", "中度 (Moderate)", "轻度 (Minor)"],
    # 评估时关注的临床维度
    "clinical_dimensions": ["相互作用机制", "临床后果", "处置建议", "严重程度"],
}


# ============================================================
# 1. 黄金数据集构建 —— 药物相互作用专用
# ============================================================
def generate_drug_golden_dataset(num_samples: int = 20,
                                  output_file: str = "drug_golden_dataset.json") -> bool:
    """
    从向量库随机抽取药物相关文本块，利用 LLM 生成：
      - 药物相互作用查询问题
      - 标准答案（含机制 + 严重程度 + 处置建议）
      - 源文本片段（用于检索命中验证）
    """
    print(f"\n[Dataset] 正在构建药物相互作用黄金数据集 (目标: {num_samples} 条)...")

    chroma = Chroma(
        collection_name=config.collection_name,
        embedding_function=config.chosen_embedding_model,
        persist_directory=config.persist_directory,
    )
    all_data = chroma.get()
    total_docs = len(all_data["ids"])

    if total_docs == 0:
        print("[Error] 向量库为空，请先上传药物说明书 / 药典 / 文献 PDF！")
        return False

    indices = random.sample(range(total_docs), min(num_samples, total_docs))
    llm = config.chosen_chat_model

    # ── 药物相互作用专用生成 Prompt ──────────────────────────────
    generator_prompt = ChatPromptTemplate.from_template(
        """You are a clinical pharmacology expert building an evaluation dataset.

Based on the following drug-related text chunk, generate ONE realistic clinical query
about drug-drug interactions that can be answered using ONLY this text.

Requirements for the question:
- Mention at least one specific drug name
- Ask about interaction mechanism, severity, or clinical management
- Be phrased as a pharmacist or physician would ask (e.g. "Can I co-administer X with Y?")

Also provide:
- ground_truth: concise clinical answer (mechanism + severity + recommendation)
- severity: one of [Contraindicated, Major, Moderate, Minor, Unknown]
- drug_names: list of drug names mentioned

Text Chunk:
{context}

Output JSON only (no markdown):
{{
    "question": "...",
    "ground_truth": "...",
    "severity": "...",
    "drug_names": ["drug_a", "drug_b"]
}}
"""
    )

    dataset = []
    success_count = 0

    for i, idx in enumerate(indices):
        doc_text = all_data["documents"][idx]
        doc_id = all_data["ids"][idx]

        try:
            chain = generator_prompt | llm | JsonOutputParser()
            qa = chain.invoke({"context": doc_text})

            dataset.append({
                "id": f"drug_test_{i:03d}",
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "severity": qa.get("severity", "Unknown"),
                "drug_names": qa.get("drug_names", []),
                "source_text_preview": doc_text[:150],
                "full_context": doc_text,
                "source_doc_id": doc_id,
            })
            success_count += 1
            print(f"  [{success_count:>2}/{num_samples}] {qa['question'][:60]}...")

        except Exception as e:
            print(f"  [Skip] 生成失败 (idx={idx}): {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\n[Dataset] ✅ 成功构建 {success_count} 条，已保存至 {output_file}")
    return success_count > 0


# ============================================================
# 2. 检索性能评估 —— 药物相互作用场景
# ============================================================
def evaluate_drug_retrieval(dataset_path: str = "drug_golden_dataset.json",
                             top_k: int = 3) -> dict:
    """
    评估检索器在药物相互作用查询上的表现。

    指标：
      - Hit Rate @ K : 正确文档出现在前 K 个结果中的比例
      - MRR          : Mean Reciprocal Rank（越高越好，最大为 1）
      - Drug Name Hit: 检索结果中至少包含查询涉及药物名称的比例（业务专项）
    """
    if not os.path.exists(dataset_path):
        print(f"[Error] 数据集 {dataset_path} 不存在，请先生成。")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    vector_service = VectorStoreService(embedding=config.chosen_embedding_model)
    retriever = vector_service.vector_store.as_retriever(search_kwargs={"k": top_k})

    hits = 0
    mrr_sum = 0.0
    drug_name_hits = 0

    print(f"\n[Retrieval Eval] Top-K={top_k} | 共 {len(test_cases)} 条测试...")

    for case in test_cases:
        query = case["question"]
        target_snippet = case["source_text_preview"]
        query_drug_names = [d.lower() for d in case.get("drug_names", [])]

        retrieved_docs = retriever.invoke(query)

        # ① 原文命中
        hit_rank = -1
        for rank, doc in enumerate(retrieved_docs):
            if target_snippet in doc.page_content:
                hit_rank = rank + 1
                break

        if hit_rank > 0:
            hits += 1
            mrr_sum += 1.0 / hit_rank

        # ② 药物名称覆盖（业务专项：检索结果是否含查询药物）
        retrieved_text = " ".join(d.page_content.lower() for d in retrieved_docs)
        if any(drug in retrieved_text for drug in query_drug_names):
            drug_name_hits += 1

    total = len(test_cases)
    result = {
        "k": top_k,
        "hit_rate": round(hits / total, 4),
        "mrr": round(mrr_sum / total, 4),
        "drug_name_coverage": round(drug_name_hits / total, 4),
    }

    print(f"  → Hit Rate @ {top_k}      : {result['hit_rate']:.2%}")
    print(f"  → MRR                  : {result['mrr']:.4f}")
    print(f"  → Drug Name Coverage   : {result['drug_name_coverage']:.2%}")

    return result


# ============================================================
# 3. 生成质量评估 —— 临床维度 LLM-as-a-Judge
# ============================================================
class DrugInteractionGrade(BaseModel):
    faithfulness: int = Field(
        description="0 or 1. Is the answer derived solely from retrieved context?"
    )
    correctness: int = Field(
        description="0 or 1. Does the answer match the ground truth clinically?"
    )
    severity_match: int = Field(
        description="0 or 1. Does the predicted severity level match ground truth severity?"
    )
    has_mechanism: int = Field(
        description="0 or 1. Does the answer explain the pharmacological mechanism?"
    )
    has_recommendation: int = Field(
        description="0 or 1. Does the answer include a clinical management recommendation?"
    )
    safety_flag: int = Field(
        description="0 or 1. Does the answer correctly flag high-risk interactions (Contraindicated/Major)?"
    )
    reason: str = Field(description="Brief clinical reasoning for the scores")


def evaluate_drug_generation(dataset_path: str = "drug_golden_dataset.json",
                               limit: int = 5) -> dict:
    """
    使用 LLM 裁判对药物相互作用回答进行多维临床评分：
      - Faithfulness       : 回答是否来自检索上下文
      - Correctness        : 与标准答案语义一致性
      - Severity Match     : 严重程度分级是否正确（安全关键）
      - Has Mechanism      : 是否解释了药理机制
      - Has Recommendation : 是否给出处置建议
      - Safety Flag        : 高危交互是否正确标记
    """
    print(f"\n[Generation Eval] LLM-as-a-Judge | 评估前 {limit} 条...")

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    rag_service = RagService()
    eval_llm = config.chosen_chat_model
    vector_service = VectorStoreService(embedding=config.chosen_embedding_model)

    # ── 药物专用评分 Prompt ──────────────────────────────────────
    grade_prompt = ChatPromptTemplate.from_template(
        """You are a senior clinical pharmacist grading an AI drug interaction assistant.

[Retrieved Context]:
{context}

[Clinical Query]:
{question}

[Ground Truth Answer]:
{ground_truth}
[Ground Truth Severity]: {severity}

[AI-Generated Answer]:
{ai_answer}

Grade each dimension strictly (1=Pass, 0=Fail):
1. faithfulness      - Answer uses only context, no hallucination
2. correctness       - Clinical content matches ground truth
3. severity_match    - Severity level matches ground truth severity
4. has_mechanism     - Explains pharmacological mechanism (PK/PD)
5. has_recommendation - Provides actionable clinical management advice
6. safety_flag       - Correctly warns if interaction is Contraindicated or Major

Return JSON only (no markdown):
{{
    "faithfulness": 0_or_1,
    "correctness": 0_or_1,
    "severity_match": 0_or_1,
    "has_mechanism": 0_or_1,
    "has_recommendation": 0_or_1,
    "safety_flag": 0_or_1,
    "reason": "brief explanation"
}}
"""
    )

    score_keys = ["faithfulness", "correctness", "severity_match",
                  "has_mechanism", "has_recommendation", "safety_flag"]
    scores = {k: [] for k in score_keys}
    detail_rows = []

    for case in test_cases[:limit]:
        question = case["question"]
        ground_truth = case["ground_truth"]
        severity = case.get("severity", "Unknown")

        # ① RAG 生成回答
        try:
            ai_answer = rag_service.chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": "drug_eval_session"}},
            )
        except Exception as e:
            print(f"  [Skip] RAG 调用失败: {e}")
            continue

        # ② 独立检索上下文（供裁判使用）
        retriever = vector_service.get_retriever()
        docs = retriever.invoke(question)
        context_str = "\n\n---\n\n".join(d.page_content for d in docs)

        # ③ LLM 打分
        grader_chain = grade_prompt | eval_llm | JsonOutputParser()
        try:
            grade = grader_chain.invoke({
                "context": context_str,
                "question": question,
                "ground_truth": ground_truth,
                "severity": severity,
                "ai_answer": ai_answer,
            })

            for k in score_keys:
                scores[k].append(int(grade.get(k, 0)))

            row = {
                "question": question[:40] + "...",
                "severity": severity,
                **{k: grade.get(k, "-") for k in score_keys},
                "reason": grade.get("reason", "")[:60],
            }
            detail_rows.append(row)

            flag_str = "⚠️ 安全标记" if grade.get("safety_flag") == 1 else "  "
            print(
                f"  {flag_str} Q: {question[:35]:<35} | "
                f"Faith:{grade['faithfulness']} Corr:{grade['correctness']} "
                f"Sev:{grade['severity_match']} Mech:{grade['has_mechanism']} "
                f"Rec:{grade['has_recommendation']} Safe:{grade['safety_flag']}"
            )

        except Exception as e:
            print(f"  [Error] 打分失败: {e}")

    # ── 汇总 ────────────────────────────────────────────────────
    avg_scores = {}
    for k in score_keys:
        avg_scores[k] = round(sum(scores[k]) / len(scores[k]), 3) if scores[k] else 0.0

    print("\n📊 药物相互作用生成质量评估结果:")
    df = pd.DataFrame([avg_scores])
    print(df.to_string(index=False))

    # 输出详细明细
    if detail_rows:
        print("\n📋 逐条评分明细:")
        df_detail = pd.DataFrame(detail_rows)
        print(df_detail[["question", "severity"] + score_keys].to_string(index=False))

    return avg_scores


# ============================================================
# 4. A/B 测试主流程
# ============================================================
def run_drug_ab_testing_pipeline():
    print("=" * 65)
    print("💊  药物相互作用检索 RAG Pipeline — 工业级量化评估")
    print("    Drug Interaction Retrieval A/B Evaluation Framework")
    print("=" * 65)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_file = "drug_golden_dataset.json"

    # ── Step 1: 数据集 ───────────────────────────────────────────
    print(f"\n[Step 1/3] 数据集检查...")
    if not os.path.exists(dataset_file):
        print("  数据集未找到，开始自动生成...")
        ok = generate_drug_golden_dataset(num_samples=15, output_file=dataset_file)
        if not ok:
            print("  ❌ 数据集生成失败，终止评估。")
            return
    else:
        with open(dataset_file) as f:
            existing = json.load(f)
        print(f"  ✅ 已加载现有数据集，共 {len(existing)} 条测试用例。")

    # ── Step 2: 检索 A/B 测试 ────────────────────────────────────
    print("\n[Step 2/3] 检索 A/B 测试 (A: Top-K=2  vs  B: Top-K=5)")
    configs_to_test = [
        {"top_k": 2, "label": "A (Top-2, 精确)"},
        {"top_k": 5, "label": "B (Top-5, 召回)"},
    ]
    retrieval_results = []
    for cfg in configs_to_test:
        print(f"\n  --- 配置 {cfg['label']} ---")
        res = evaluate_drug_retrieval(dataset_path=dataset_file, top_k=cfg["top_k"])
        res["label"] = cfg["label"]
        retrieval_results.append(res)

    print("\n📊 检索 A/B 对比表:")
    df_r = pd.DataFrame(retrieval_results).set_index("label")
    df_r.columns = ["Top-K", "Hit Rate", "MRR", "Drug Name Coverage"]
    print(df_r.to_string())

    # 给出推荐
    best = max(retrieval_results, key=lambda x: x["hit_rate"] * 0.5 + x["drug_name_coverage"] * 0.5)
    print(f"\n  🏆 综合推荐配置: {best['label']}")

    # ── Step 3: 端到端生成质量评估 ───────────────────────────────
    print("\n[Step 3/3] 端到端生成质量评估 (LLM-as-a-Judge, 限 3 条省 Token)")
    gen_scores = evaluate_drug_generation(dataset_path=dataset_file, limit=3)

    # ── 最终报告 ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📈  最终评估报告摘要")
    print("=" * 65)
    print(f"  检索 Hit Rate @ 5    : {best['hit_rate']:.2%}")
    print(f"  检索 MRR             : {best['mrr']:.4f}")
    print(f"  药物名称覆盖率       : {best['drug_name_coverage']:.2%}")
    print(f"  生成 Faithfulness    : {gen_scores.get('faithfulness', 0):.2f}")
    print(f"  生成 Correctness     : {gen_scores.get('correctness', 0):.2f}")
    print(f"  严重程度分级准确率   : {gen_scores.get('severity_match', 0):.2f}  ← 安全关键指标")
    print(f"  机制解释覆盖率       : {gen_scores.get('has_mechanism', 0):.2f}")
    print(f"  处置建议覆盖率       : {gen_scores.get('has_recommendation', 0):.2f}")
    print(f"  高危交互安全标记率   : {gen_scores.get('safety_flag', 0):.2f}  ← 安全关键指标")
    print("=" * 65)

    # 保存 JSON 报告
    report = {
        "timestamp": timestamp,
        "business_scenario": "drug_interaction_retrieval",
        "retrieval_ab_results": retrieval_results,
        "generation_scores": gen_scores,
        "recommended_retrieval_config": best["label"],
    }
    report_path = f"drug_eval_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\n  💾 详细报告已保存至 {report_path}")
    print("\n✅ 评估完成。以上量化指标可直接用于：")
    print("   · 技术报告与模型卡片")
    print("   · 产品上线前的质量 Gate")


# ============================================================
if __name__ == "__main__":
    run_drug_ab_testing_pipeline()