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

        self.vector_space = VectorStoreService(embedding= config.chosen_embedding_model)
        self.retriever = self.vector_space.get_retriever()

        self.prompt_template = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "你是一位专业的临床药师 AI 助手。你的职责是根据提供的药物数据库信息，为用户提供准确的用药安全咨询。\n\n"
                    "回答要求：\n"
                    "1. 结构化输出：明确列出每对药物组合的相互作用\n"
                    "2. 严重程度标注：对每个相互作用标注 severity（Major/Moderate/Minor）\n"
                    "3. 机制解释：用通俗语言解释为什么会发生相互作用\n"
                    "4. 临床建议：给出实际可操作的用药建议\n"
                    "5. 诚实原则：如果数据库中没有某对药物的相互作用记录，明确说明“未找到记录”，绝不编造\n"
                    "6. 免责声明：提醒用户本系统仅供参考，具体用药请遵医嘱\n\n"
                    "数据来源说明：\n"
                    "- 【精确查询结果】来自 DrugBank 数据库的结构化记录，可信度高\n"
                    "- 【语义检索结果】来自药理学文本的语义匹配，用于补充机制解释\n"
                    "- 如果两者有矛盾，以精确查询结果为准\n\n"
                    "请基于以下信息回答：\n{context}"
                ),
                MessagesPlaceholder(variable_name="history"), #历史消息占位符
                ("human","My question is {question}")
        ])
        self.chat_model = config.chosen_chat_model

        self.chain = self.__get_chain()


    def _format_sql_context(self, sql_result: dict) -> str:
        identified = sql_result.get("identified_drugs", [])
        interactions = sql_result.get("interactions", [])
        food_list = sql_result.get("food_interactions", [])

        lines = ["=== 精确查询结果（来自 SQLite） ===", ""]

        lines.append("【识别到的药物】")
        if identified:
            for idx, d in enumerate(identified, start=1):
                name = d.get("name")
                std = d.get("standard_name") or "未匹配"
                dbid = d.get("drugbank_id") or "None"
                lines.append(f"{idx}. {name} ({std}, {dbid})")
        else:
            lines.append("- 未识别到药物名")
        lines.append("")

        lines.append("【药物相互作用】")
        if interactions:
            for row in interactions[:30]:
                a_name = row.get("drug_a_name") or row.get("drug_a_id")
                b_name = row.get("drug_b_name") or row.get("drug_b_id")
                severity = (row.get("severity") or "unknown").capitalize()
                desc = row.get("description") or "无描述"
                lines.append(f"- {a_name} × {b_name}")
                lines.append(f"  严重程度：{severity}")
                lines.append(f"  描述：{desc}")
        else:
            lines.append("- 未找到记录")
        lines.append("")

        lines.append("【食物禁忌】")
        if food_list:
            for row in food_list[:30]:
                drug_name = row.get("drug_name") or row.get("drug_id")
                desc = row.get("description") or "无描述"
                lines.append(f"- {drug_name}：{desc}")
        else:
            lines.append("- 未找到记录")
        lines.append("")

        msg = sql_result.get("message")
        if msg:
            lines.append(f"【检索备注】{msg}")
            lines.append("")

        return "\n".join(lines)


    def _format_vector_context(self, docs: list[Document]) -> str:
        lines = ["=== 语义检索结果（来自 ChromaDB） ===", ""]
        if not docs:
            lines.append("未检索到相关语义文档。")
            return "\n".join(lines)

        for idx, doc in enumerate(docs, start=1):
            field = doc.metadata.get("field", "unknown")
            drug_name = doc.metadata.get("drug_name", "unknown")
            content = (doc.page_content or "").strip()
            lines.append(f"[{idx}] {drug_name}（{field}）：{content}")
        return "\n".join(lines)


    def retrieve(self, user_input: str) -> str:
        """
        双路召回：
        1) SQLite 精确查询
        2) ChromaDB 语义检索
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
                "message": f"SQLite 检索失败：{exc}",
            }

        try:
            docs = self.retriever.invoke(user_input)
        except Exception as exc:
            docs = [Document(page_content=f"向量检索失败：{exc}", metadata={"field": "error"})]

        sql_context = self._format_sql_context(sql_result)
        vec_context = self._format_vector_context(docs)
        return f"{sql_context}\n\n{vec_context}"


    def __get_chain(self):
        chain = (
            {
                "question": itemgetter("question"), 
                "context": itemgetter("question") | RunnableLambda(self.retrieve),
                "history": itemgetter("history")

            }
            | self.prompt_template
            | self.chat_model | StrOutputParser()
        )
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="question", #把input_messages放进input里，使用的key是"question"
            history_messages_key="history" #把历史消息history_messages放进input里，使用的key是"history"
        )
        return conversation_chain


if __name__ == "__main__":

    rag_service = RagService()
    
    response = rag_service.chain.invoke(
        {"question":"input question"},
        config={"configurable":{"session_id":"user123"}}
    )
    print(response)
