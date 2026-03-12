from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

md5_path = "./md5.text"

# Chroma
collection_name = "drug_rag"
persist_directory = "./chroma_db"
sqlite_db_path = "./drug_structured.db"

# Splitter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", ";", ","," ", ""]

minimum_splitter_size = 1000
num_of_matched_docs = 2
sql_ner_model_name = "gpt-4o-mini"

chosen_embedding_model = OpenAIEmbeddings(model ="text-embedding-3-large")
chosen_chat_model = ChatOpenAI(model="gpt-4o-mini")

session_config = {
    "configurable":{
        "session_id":"user_001",
    }
}