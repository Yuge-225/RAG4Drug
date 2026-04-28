import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage,message_to_dict,messages_from_dict
import json

"""
聊天历史管理类：负责把聊天记录持久化存储到本地文件中，让不同session的对话可以被保存，读取，清空
    每个session id 对应一个文件
    聊天信息不是存在内存里，而是存在于JSON文件
    该类实现了LangChain规定的聊天历史接口
"""

def get_history(session_id):
    """
    根据session_id创建一个FileChatMessageHistory对象,以后LangChain就用它来存储和读取对话
    """
    return FileChatMessageHistory(session_id = session_id,storage_path = "./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id  # 会话id
        self.storage_path = storage_path # 不同会话id的存储文件所放在的文件夹路径

        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id) # path = "./chat_history/session_id"

        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok = True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # Sequence序列
        all_messages = list(self.messages) # self.messages调用的是@property,实际上会从持久化文件里读取出消息
        all_messages.extend(messages) # 新的消息列表和已有的消息列表融合成一个list

        # 将数据同步写入到本地文件夹中
        # 写入时，需要使用message_to_dict将消息格式从list[BaseMessage]转化为Json数据格式
        new_messages = [message_to_dict(message) for message in all_messages]

        # 将Json格式的数据写入文件
        with open(self.file_path,"w", encoding="utf-8") as f:
            json.dump(new_messages,f) # 整个历史被完整覆盖写入

    @property # @property修饰器将messages方法变成成员属性使用
    def messages(self) -> list[BaseMessage]:
        """
        从self.file_path读取数据
        """
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages_data = json.load(f) #加载Json格式数据
                return messages_from_dict(messages_data) #Json格式数据 --> list[BaseMessage]
        except FileNotFoundError:
            return []
        
    def clear(self) -> None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f) #将文件内容变成空数组

        

