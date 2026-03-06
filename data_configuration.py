from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os

md5_path = "./md5.text"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# Splitter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", ";", ","," ", ""]

minimum_splitter_size = 1000
num_of_matched_docs = 2

chosen_embedding_model = OpenAIEmbeddings(model ="text-embedding-3-large")
chosen_chat_model = ChatOpenAI(model="gpt-4o-mini")

session_config = {
    "configurable":{
        "session_id":"user_001",
    }
}