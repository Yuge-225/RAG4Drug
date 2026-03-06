from vector_store import VectorStoreService
import data_configuration as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from file_history_store import get_history

class RagService():
    def __init__(self):

        self.vector_space = VectorStoreService(embedding= config.chosen_embedding_model)

        self.prompt_template = ChatPromptTemplate.from_messages([
                ("system","You are an excellent medical analyst, Please help me with the task using the following information: \n {context}"),
                MessagesPlaceholder(variable_name="history"), #历史消息占位符
                ("human","My question is {question}")
        ])
        self.chat_model = config.chosen_chat_model

        self.chain = self.__get_chain()


    def __get_chain(self):
        retriever = self.vector_space.get_retriever() # Retriever是一个Runnable
        """
        chain的数据流
        1. retriever接受question,检索到最相关的k个Documents,返回list[Documents]
        2. 相关文档documents被填入到prompt的{context}中
        3. question & company_name被直接传递
        """
        def format_document(docs: list[Document]) -> str:
            if not docs:
                return "No Related Information Retrieved from Vector Databse!"
            else:
                format_str = ""
                for doc in docs:
                    format_str += f"Document Info: {doc.page_content} \n Document Metadata: {doc.metadata} \n\n"
                return format_str
            
        chain = (
            {
                "question": itemgetter("question"), 
                "context": itemgetter("question")| retriever | format_document,
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
