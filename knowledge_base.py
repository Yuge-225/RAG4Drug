import data_configuration 
import os
import hashlib
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import random


def check_md5(md5_str):
    """
    检查传入的md5_str是否存在于md5.text文件中
    """
    if os.path.exists(data_configuration.md5_path):
        # md5.text存在，逐行检查传入的md5_str是否能被匹配到
        for line in open(data_configuration.md5_path,"r",encoding="utf-8").readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False

    else:
        # md5.text不存在，先创建该文件
        open(data_configuration.md5_path,"w",encoding="utf-8").close
        return False
    

def save_md5(md5_str):
    """
    将md5_str传入到md5.text文件中
    """
    with open(data_configuration.md5_path,'a',encoding="utf-8") as f:
        f.write(md5_str+"\n")
        

def get_string_md5(input_str,encoding = "utf-8"):
    """
    将字符串转化为md5格式数据
    """
    
    bytes_rep = input_str.encode(encoding) # 将字符串以utf-8的编码形式转为二进制字节码

    md5_obj = hashlib.md5() # 创建md5对象
    md5_obj.update(bytes_rep) # 将二进制字节码传入到md5对象中
    md5_hex = md5_obj.hexdigest() # 转化为16进制的md5格式
    return md5_hex

    


class KnowledgeBaseService:
    def __init__(self):
        
        self.chroma = Chroma(
            collection_name = data_configuration.collection_name, # 向量数据库的表名
            embedding_function= OpenAIEmbeddings(model ="text-embedding-3-large" ), #嵌入模型
            persist_directory = data_configuration.persist_directory # 数据库本地存储文件夹
        ) #Chroma向量库对象

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = data_configuration.chunk_size, #分割后的文本段最大长度
            chunk_overlap = data_configuration.chunk_overlap, #连续文本之间的字符重叠数
            separators = data_configuration.separators, # 自然段落划分的符号
            length_function = len, #使用Python自带的len函数做长度统计
        )
    
    def upload_by_str(self,data:str,filename):
        """
        将传入的字符串做md5处理，检查是否已经存在于向量数据库中
            若存在：不做添加（去重）
            若不存在：保存并添加到向量数据库中
        """

        data_md5 = get_string_md5(input_str=data)

        if check_md5(data_md5):
            return f"Status [SKIP] Current File have been processed before"
        
        if len(data) > data_configuration.minimum_splitter_size:
            data_chunks = self.splitter.split_text(data)
        else:
            data_chunks = [data]
        
        metadata = {
            "filename":filename,
            "author":"Yuge",
            "file_create time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        metadatas = [metadata] * len(data_chunks)
        self.chroma.add_texts(texts=data_chunks, metadatas=metadatas)
        
        save_md5(data_md5)
        return f"Status [SUCCESS] Current File Saved To Vector Database"
                
    def get_random_chunks(self, sample_size=3):
        """
        随机从向量数据库中抽取 sample_size 个切片用于检查
        """
        # 获取所有数据的 ID (只拿 ID 和 Document，不拿 Embedding 以节省带宽)
        collection_data = self.chroma.get(include=['documents', 'metadatas'])
        
        total_chunks = len(collection_data['ids'])
        
        if total_chunks == 0:
            return []
            
        # 确保抽样数量不超过总数
        k = min(sample_size, total_chunks)
        
        # 生成随机索引
        random_indices = random.sample(range(total_chunks), k)
        
        results = []
        for idx in random_indices:
            results.append({
                "id": collection_data['ids'][idx],
                "content": collection_data['documents'][idx],
                "metadata": collection_data['metadatas'][idx]
            })
        return results

    def get_database_status(self):
        """
        获取向量数据库的当前统计信息
        """
        # 获取所有数据（仅获取元数据以节省内存）
        collection_data = self.chroma.get()
        
        total_chunks = len(collection_data['ids'])
        
        # 统计唯一文件数
        filenames = set()
        for meta in collection_data['metadatas']:
            if meta and 'filename' in meta:
                filenames.add(meta['filename'])
        
        return {
            "total_chunks": total_chunks,
            "total_files": len(filenames),
            "file_list": list(filenames)
        }
    
    def clear_database(self):
        """
        (可选) 增加一个清空数据库的功能，方便测试
        """
        self.chroma.delete_collection()
        # 重新初始化以便下次使用
        self.chroma = Chroma(
            collection_name = data_configuration.collection_name, 
            embedding_function= OpenAIEmbeddings(model ="text-embedding-3-large" ),
            persist_directory = data_configuration.persist_directory
        )
        # 清空 md5 记录
        open(data_configuration.md5_path, "w", encoding="utf-8").close()
        return "Database Cleared"