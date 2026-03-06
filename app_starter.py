import time
from rag import RagService
import streamlit as st
import data_configuration as config

st.title("Medicine Use Assistant")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = [{
        "role":"assistant",
        "content":"Hi, I am your medicine use analyst. How can I help you?"
    }
    ]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list =[]
    with st.spinner("AI Inferencing..."):
        #从RAG拿到一个流式生成器，rag_stream并非字符串，它是一个generator，会一点一点yield内容
        res_stream = st.session_state["rag"].chain.stream({"question":prompt}, config.session_config) 

        def capture(generator, cache_list):
            # generator：原始的流式输出
            # cache_list：用来缓存所有输出chunk的列表
            for chunk in generator:
                # chunk：对于模型生成的每一个token片段而言
                cache_list.append(chunk) # 先存起来，用于后面拼接回答
                yield chunk # 再yield出去，让streamlit实时显示
        
        with st.chat_message("assistant"):
            st.write_stream(
                capture(res_stream,ai_res_list)
            )

        st.session_state["message"].append({"role":"assistant", "content":"".join(ai_res_list)})