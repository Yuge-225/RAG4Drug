import streamlit as st
from knowledge_base import KnowledgeBaseService
import pandas as pd # 如果没装 pandas，可以把相关展示代码改成 st.write
import time
st.set_page_config(page_title="RAG Data Manager", layout="wide")

st.title("📂 RAG Knowledge Base Manager")

# 初始化 Service
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

# ==========================================
# 1. 侧边栏：数据库监控 (Dashboard)
# ==========================================
with st.sidebar:
    st.header("📊 Database Status")
    
    # 在侧边栏增加一个抽查区域
    st.subheader("🕵️ Data Inspection")
    
    if st.button("🎲 Random Peek (3 Chunks)"):
        with st.spinner("Fetching random chunks..."):
            random_chunks = st.session_state["service"].get_random_chunks(sample_size=3)
            
            if not random_chunks:
                st.warning("Database is empty.")
            else:
                # 使用模态框 (Dialog) 或者在主界面显示
                # 这里为了简单，直接在侧边栏下方或者用 st.expander 显示
                st.success(f"Found {len(random_chunks)} chunks:")
                
                for i, chunk in enumerate(random_chunks):
                    # 获取文件名（防止 metadata 为空的情况）
                    meta = chunk['metadata'] or {}
                    filename = meta.get('filename', 'Unknown File')
                    
                    with st.expander(f"Chunk #{i+1} from {filename}", expanded=True):
                        st.markdown(f"**ID:** `{chunk['id']}`")
                        st.caption(f"**Metadata:** {meta}")
                        st.text_area("Content Preview", chunk['content'], height=150, disabled=True)
    # 获取最新状态
    stats = st.session_state["service"].get_database_status()
    
    # 显示核心指标
    col1, col2 = st.columns(2)
    col1.metric("Files", stats["total_files"])
    col2.metric("Chunks", stats["total_chunks"])
    
    st.divider()
    
    # 显示文件列表
    st.subheader("Stored Documents")
    if stats["file_list"]:
        st.markdown("\n".join([f"- 📄 {f}" for f in stats["file_list"]]))
    else:
        st.info("Database is empty.")
        
    st.divider()
    
    # (危险操作) 清空数据库按钮
    if st.button("🗑️ Reset Database", type="primary"):
        res = st.session_state["service"].clear_database()
        st.toast(res)
        st.rerun() # 刷新页面

# ==========================================
# 2. 主区域：文件上传
# ==========================================
st.subheader("Upload New Documents")

# 支持 PDF 和 TXT
file_uploader = st.file_uploader(
    label="Upload Financial Reports (PDF/TXT)",
    type=["txt", "pdf"],
    accept_multiple_files=False
)

if file_uploader is not None:
    file_name = file_uploader.name
    file_size = file_uploader.size
    
    # 显示文件基本信息
    st.info(f"File: **{file_name}** ({file_size / 1024:.2f} KB)")
    
    # 提取文本 (这里简化了之前的 PDF 逻辑，记得把你上一轮改好的 PDF 解析逻辑放进来)
    import pypdf
    text = ""
    
    # 使用 st.status 提供更好的上传反馈
    with st.status("Processing File...", expanded=True) as status:
        
        # Step 1: 解析
        st.write("📖 Parsing content...")
        try:
            if file_uploader.type == "application/pdf":
                pdf_reader = pypdf.PdfReader(file_uploader)
                text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            else:
                text = file_uploader.getvalue().decode("utf-8")
                
            st.write(f"✅ Extracted {len(text)} characters.")
            
            # Step 2: 向量化存储
            st.write("🧠 Vectorizing & Storing...")
            if text:
                res = st.session_state["service"].upload_by_str(data=text, filename=file_name)
                
                status.update(label="Processing Complete!", state="complete", expanded=False)
                
                # Step 3: 结果展示
                if "SUCCESS" in res:
                    st.success(f"Upload Success: {file_name}")
                    st.balloons() # 成功撒花效果
                    # 强制刷新页面以更新侧边栏统计数据
                    time.sleep(1)
                    st.rerun()
                elif "SKIP" in res:
                    st.warning("File already exists (Skipped by MD5 check).")
            else:
                status.update(label="Extraction Failed", state="error")
                st.error("No text could be extracted from this file.")
                
        except Exception as e:
            status.update(label="Error Occurred", state="error")
            st.error(f"Error: {e}")
