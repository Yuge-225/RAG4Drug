from operator import itemgetter

import data_configuration as config
from drug_sql_retriever import retrieve_from_sql
from file_history_store import get_history
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from vector_store import VectorStoreService

class RagService():
    def __init__(self):

        self.vector_space = VectorStoreService(embedding=config.chosen_embedding_model)
        self.retriever = self.vector_space.get_retriever()

        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a professional clinical pharmacist AI assistant. "
                "Your role is to provide accurate drug safety consultation based on the drug database information provided.\n\n"
                "Response requirements:\n"
                "1. Structured output: clearly list the interaction for each drug pair\n"
                "2. Severity labeling: label each interaction with its severity (Major/Moderate/Minor)\n"
                "3. Mechanism explanation: explain in plain language why the interaction occurs\n"
                "4. Clinical advice: provide actionable medication recommendations\n"
                "5. Honesty principle: if no interaction record exists for a drug pair, clearly state 'No record found' — never fabricate\n"
                "6. Disclaimer: remind the user this system is for reference only; follow your doctor's advice for actual medication\n\n"
                "Data source notes:\n"
                "- [Exact Query Results] are structured records from the DrugBank database — high reliability\n"
                "- [Semantic Search Results] are semantically matched pharmacological texts — used to supplement mechanism explanations\n"
                "- If the two sources conflict, the Exact Query Results take precedence\n\n"
                "Answer based on the following information:\n{context}"
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "My question is {question}")
        ])
        self.chat_model = config.chosen_chat_model
        self.chain = self.__get_chain()

    def _format_sql_context(self, sql_result: dict) -> str:
        identified = sql_result.get("identified_drugs", [])
        interactions = sql_result.get("interactions", [])
        food_list = sql_result.get("food_interactions", [])

        lines = ["=== Exact Query Results (SQLite) ===", ""]

        lines.append("[Identified Drugs]")
        if identified:
            for idx, d in enumerate(identified, start=1):
                name = d.get("name")
                std = d.get("standard_name") or "not matched"
                dbid = d.get("drugbank_id") or "None"
                lines.append(f"{idx}. {name} ({std}, {dbid})")
        else:
            lines.append("- No drug names identified")
        lines.append("")

        lines.append("[Drug Interactions]")
        if interactions:
            for row in interactions[:30]:
                a_name = row.get("drug_a_name") or row.get("drug_a_id")
                b_name = row.get("drug_b_name") or row.get("drug_b_id")
                severity = (row.get("severity") or "unknown").capitalize()
                desc = row.get("description") or "No description"
                lines.append(f"- {a_name} x {b_name}")
                lines.append(f"  Severity: {severity}")
                lines.append(f"  Description: {desc}")
        else:
            lines.append("- No records found")
        lines.append("")

        lines.append("[Food Interactions]")
        if food_list:
            for row in food_list[:30]:
                drug_name = row.get("drug_name") or row.get("drug_id")
                desc = row.get("description") or "No description"
                lines.append(f"- {drug_name}: {desc}")
        else:
            lines.append("- No records found")
        lines.append("")

        msg = sql_result.get("message")
        if msg:
            lines.append(f"[Retrieval Note] {msg}")
            lines.append("")

        return "\n".join(lines)

    def _format_vector_context(self, docs: list[Document]) -> str:
        lines = ["=== Semantic Search Results (ChromaDB) ===", ""]
        if not docs:
            lines.append("No relevant semantic documents retrieved.")
            return "\n".join(lines)

        for idx, doc in enumerate(docs, start=1):
            field = doc.metadata.get("field", "unknown")
            drug_name = doc.metadata.get("drug_name", "unknown")
            content = (doc.page_content or "").strip()
            lines.append(f"[{idx}] {drug_name} ({field}): {content}")
        return "\n".join(lines)

    def retrieve(self, user_input: str) -> str:
        """
        Dual-path retrieval:
        1) SQLite exact query
        2) ChromaDB semantic search
        """
        try:
            sql_result = retrieve_from_sql(user_input)
        except Exception as exc:
            sql_result = {
                "identified_drugs": [],
                "interactions": [],
                "food_interactions": [],
                "dosages": [],
                "enzymes": [],
                "message": f"SQLite retrieval failed: {exc}",
            }

        try:
            docs = self.retriever.invoke(user_input)
        except Exception as exc:
            docs = [Document(page_content=f"Vector retrieval failed: {exc}", metadata={"field": "error"})]

        self.last_sql_result = sql_result
        self.last_vector_docs = docs

        sql_context = self._format_sql_context(sql_result)
        vec_context = self._format_vector_context(docs)
        return f"{sql_context}\n\n{vec_context}"

    def __get_chain(self):
        chain = (
            {
                "question": itemgetter("question"),
                "context":  itemgetter("question") | RunnableLambda(self.retrieve),
                "history":  itemgetter("history"),
            }
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        return conversation_chain


if __name__ == "__main__":
    rag_service = RagService()
    response = rag_service.chain.invoke(
        {"question": "Can warfarin and aspirin be taken together?"},
        config={"configurable": {"session_id": "user123"}},
    )
    print(response)
