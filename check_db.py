# check_db.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import data_configuration as config

def check_database():
    print("正在连接向量数据库...")
    
    # 初始化数据库连接
    chroma = Chroma(
        collection_name=config.collection_name,
        embedding_function=config.chosen_embedding_model,
        persist_directory=config.persist_directory
    )
    
    # 获取所有数据（仅元数据，不加载向量，速度快）
    data = chroma.get()
    
    ids = data['ids']
    metadatas = data['metadatas']
    
    # 统计
    total_chunks = len(ids)
    filenames = set()
    
    print(f"\n======== 数据库资产盘点 ========")
    print(f"总切片数 (Chunks): {total_chunks}")
    
    if total_chunks > 0:
        for meta in metadatas:
            if meta and 'filename' in meta:
                filenames.add(meta['filename'])
                
        print(f"包含文件数 (Files): {len(filenames)}")
        print("\n已存储的文件清单:")
        for f in filenames:
            print(f"  - {f}")
            
        # 估算字符数 (假设平均每个chunk 500-1000字符)
        print(f"\n估算总文本量: ~{total_chunks * 800 / 10000:.2f} 万字符")
    else:
        print("数据库是空的！")

if __name__ == "__main__":
    check_database()