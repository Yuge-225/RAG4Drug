from langchain_chroma import Chroma
import data_configuration as config
from langchain_openai import OpenAIEmbeddings




class VectorStoreService:
    def __init__(self, embedding):
        self.embedding = embedding

        self.vector_store = Chroma(
            embedding_function = self.embedding,
            collection_name = config.collection_name,
            persist_directory = config.persist_directory
        )
    
    def get_retriever(self):
        """
        返回向量检索器,方便加入chain
        """
        return self.vector_store.as_retriever(
            search_kwargs = {"k": config.num_of_matched_docs} #参数k表示retriever将根据query，返回最相似的k个文档
        )

if __name__ =="__main__":
    vectorStoreService = VectorStoreService(OpenAIEmbeddings(model ="text-embedding-3-large" ))
    retriever = vectorStoreService.get_retriever()

    matched_docs = retriever.invoke("cash flow")
    print(matched_docs[0])
    


