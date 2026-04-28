# 💊 Drug Interaction RAG System

一个基于 RAG（检索增强生成）架构的药物相互作用智能问答系统。

输入病人正在服用的药物清单，系统会从权威药物数据库（DrugBank）中检索相关信息，由 LLM 生成结构化的用药安全报告。

---

## 目录

- [项目背景](#项目背景)
- [系统架构](#系统架构)
- [项目文件说明](#项目文件说明)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [当前进度](#当前进度)
- [注意事项](#注意事项)

---

## 项目背景

病人同时服用多种药物时，药物之间可能存在相互作用（Drug-Drug Interaction），轻则降低药效，重则危及生命。

本系统的目标是：

> 输入「病人正在服用的药物列表」→ 输出「结构化用药风险报告」

数据来源：[DrugBank](https://go.drugbank.com/)，权威药物数据库，包含约 **20,000 种药物**、**290 万条**相互作用记录。

---

## 系统架构

```
用户问题："华法林和阿司匹林可以一起吃吗？"
                    │
                    ▼
            ┌───────────────┐
            │  NER 实体识别  │  从问题中提取药物名
            │  （查同义词表）│  华法林 → DB00001
            └───────┬───────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
  ┌──────────────┐    ┌──────────────────┐
  │    SQLite    │    │     ChromaDB     │
  │  精确查询    │      │   语义向量检索    │
  │              │     │                 │
  │ • 相互作用   │      │ • 药理机制描述   │
  │ • 食物禁忌   │      │ • 适应症说明     │
  │ • 剂量信息   │      │ • 药效学信息     │
  └──────┬───────┘    └────────┬─────────┘
          └─────────┬──────────┘
                    │ 合并两路召回结果
                    ▼
            ┌───────────────┐
            │  LLM 生成报告  │
            │  (GPT-4o-mini) │
            └───────────────┘
```

**为什么需要两个数据库？**

| 数据库   | 回答什么问题                                   | 优势                   |
| -------- | ---------------------------------------------- | ---------------------- |
| SQLite   | 华法林和阿司匹林**有没有**相互作用？严重程度？ | 精确、快速、无幻觉     |
| ChromaDB | 这个相互作用的**机制是什么**？为什么会发生？   | 自然语言理解、语义检索 |

---

## 项目文件说明

```
RAG/
├── 📄 核心文件
│   ├── xml_parser.py          数据解析：把 drug_db.xml 写入 SQLite 和 ChromaDB
│   ├── rag.py                 RAG 核心：召回 + LLM 生成（待改造）
│   ├── vector_store.py        ChromaDB 向量库服务
│   ├── knowledge_base.py      知识库管理服务
│   └── data_configuration.py  全局配置（模型、路径、参数）
│
├── 🖥️ 应用界面
│   ├── app_starter.py         主聊天界面（Streamlit）
│   └── app_file_uploader.py   文件上传管理界面（暂不使用）
│
├── 🔧 工具文件
│   ├── file_history_store.py  对话历史持久化存储
│   ├── check_db.py            向量库状态检查工具
│   └── evaluation.py          RAG 评估脚本（Hit Rate、MRR、LLM-as-Judge）
│
├── 📦 数据文件（不上传 GitHub，需手动获取）
│   ├── drug_db.xml.zip        原始数据压缩包（174MB，解压后 1.77GB）
│   ├── drug_structured.db     SQLite 数据库（xml_parser.py 运行后自动生成）
│   └── chroma_db/             向量数据库目录（xml_parser.py 运行后自动生成）
│
└── 📝 运行时自动生成
    ├── chat_history/          对话历史
    └── md5.text               文件去重记录
```

---

## 环境配置

### 第一步：克隆项目

```bash
git clone https://github.com/Yuge-225/RAG4Drug
cd RAG4Drug
```

### 第二步：创建虚拟环境

```bash
conda create -n drug-rag python=3.11
conda activate drug-rag
```

> 之后每次开始工作前，都需要先激活环境：
>
> ```bash
> conda activate drug-rag
> ```

### 第三步：安装依赖

```bash
pip install streamlit \
            langchain \
            langchain-openai \
            langchain-chroma \
            langchain-text-splitters \
            langchain-core \
            chromadb \
            openai \
            pypdf \
            rich \
            pandas \
            pydantic \
            python-dotenv
```

安装完成后验证：

```bash
python -c "import streamlit, langchain, chromadb, openai; print('✅ 所有依赖安装成功')"
```

### 第四步：获取并配置 OpenAI API Key

**4.1 注册 / 登录 OpenAI**

前往 [https://platform.openai.com](https://platform.openai.com)，注册或登录账号。

**4.2 进入 API Key 管理页面**

登录后点击右上角头像 → 选择 **"API keys"**

或直接访问：[https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

**4.3 创建新的 API Key**

1. 点击 **"Create new secret key"**
2. 给 Key 起一个名字，例如 `drug-rag`
3. 点击 **"Create secret key"**
4. **立刻复制**这串 `sk-...` 字符串并保存好，弹窗关闭后将无法再次查看

**4.4 充值（新账号需要）**

前往 [https://platform.openai.com/settings/billing](https://platform.openai.com/settings/billing) 添加付款方式并充值。

本项目费用参考：

- 运行 `xml_parser.py` 解析全量数据：约 **$0.5–1.0**（一次性）
- 日常问答（GPT-4o-mini）：极低，每次问答约 **$0.001**

**4.5 在项目中配置 API Key**

在项目根目录创建 `.env` 文件：

```bash
touch .env
```

用文本编辑器打开 `.env`，写入以下内容（替换为你自己的 Key）：

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

**4.6 验证配置是否成功**

```bash
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ.get('OPENAI_API_KEY', '')
print('✅ API Key 已配置' if key.startswith('sk-') else '❌ 未找到 API Key，请检查 .env 文件')
"
```

### 第五步：准备数据文件

```bash
unzip drug_db.xml.zip
# 解压后得到 drug_db.xml（1.77GB）
```

运行解析脚本，生成 SQLite 数据库和 ChromaDB 向量库：

```bash
python xml_parser.py --xml drug_db.xml
```

预计耗时 **15–20 分钟**，运行过程中会显示实时进度。完成后自动生成：

- `drug_structured.db`（约 580MB）
- `chroma_db/` 文件夹

> 💡 如果中途中断，直接重新运行即可，脚本支持**断点续传**，会自动跳过已处理的药物，不会重复计费。

---

## 快速开始

### 启动主聊天界面

```bash
conda activate drug-rag
streamlit run app_starter.py
```

浏览器自动打开，或手动访问 `http://localhost:8501`

### 验证数据库状态

```bash
python -c "
import sqlite3
conn = sqlite3.connect('drug_structured.db')
print('药物总数:    ', conn.execute('SELECT COUNT(*) FROM drugs').fetchone()[0])
print('相互作用数:  ', conn.execute('SELECT COUNT(*) FROM drug_interactions').fetchone()[0])
print('同义词数:    ', conn.execute('SELECT COUNT(*) FROM synonyms').fetchone()[0])
conn.close()
"
```

正常输出应为：

```
药物总数:     19830
相互作用数:   2910010
同义词数:     1049999
```

---

## 当前进度

### ✅ Step 1：XML 数据解析（已完成）

从 DrugBank XML（1.77GB）解析并写入两个数据库：

**SQLite（结构化数据）**

| 表名              | 数据量       | 用途                               |
| ----------------- | ------------ | ---------------------------------- |
| drugs             | 19,830 行    | 药物基本信息（名称、半衰期、分组） |
| synonyms          | 1,049,999 行 | 所有药物别名，NER 实体识别用       |
| drug_interactions | 2,910,010 行 | 药物相互作用，RAG 核心数据         |
| food_interactions | 2,549 行     | 食物禁忌                           |
| dosages           | ~30,000 行   | 剂量信息                           |
| enzymes           | ~40,000 行   | CYP450 代谢酶信息                  |
| parse_progress    | 19,830 行    | 断点续传进度记录                   |

**ChromaDB（向量数据）**

- 共 26,980 个向量块
- 索引字段：`mechanism-of-action`、`description`、`indication`、`pharmacodynamics`
- 用于语义检索药理机制和描述文本

**相互作用严重程度分布**

```
moderate:  2,270,541 条  (78.0%)
unknown:     638,999 条  (22.0%)
severe:           54 条  ( 0.0%)
minor:           416 条  ( 0.0%)
```

> `severe` 占比偏低是已知问题，DrugBank 原文描述格式不统一导致部分严重相互作用被归类为 `unknown`，后续优化时会改进 severity 提取逻辑。

---

### ⏳ Step 2：改造 rag.py（进行中）

**目标**：将现有单路召回（仅 ChromaDB）改为双路召回（SQLite + ChromaDB）

**需要新建** `drug_sql_retriever.py`

- NER：调用 LLM 从用户输入中提取药物名 → 查 `synonyms` 表映射为 DrugBank ID
- SQL 查询：用 Drug ID 查询相互作用记录、食物禁忌、剂量信息

**需要修改** `rag.py`

- System prompt 改为临床药师角色
- 召回逻辑改为双路并行，合并后送入 LLM

---

### Step 3：系统测试

计划测试用例：

| 测试类型   | 输入示例                                   | 预期结果                       |
| ---------- | ------------------------------------------ | ------------------------------ |
| 两药联用   | "华法林和阿司匹林能一起吃吗？"             | 召回 SEVERE 级相互作用记录     |
| 三药联用   | "华法林、阿司匹林、布洛芬同时服用安全吗？" | 召回所有两两组合的相互作用     |
| 无记录边界 | "华法林和维生素C有相互作用吗？"            | 诚实回答"数据库无记录"，不编造 |

---

### Step 4：评估

复用现有 `evaluation.py`，仅需将生成问题的 prompt 替换为药物场景。

| 指标         | 含义                                   |
| ------------ | -------------------------------------- |
| Hit Rate     | 正确答案是否出现在召回结果中           |
| MRR          | 正确答案的召回排名（越靠前越好）       |
| Faithfulness | LLM 回答是否完全基于召回内容，没有幻觉 |
| Correctness  | LLM 回答与标准答案是否语义一致         |

---

### Step 5：根据评估结果优化

---

### Step 6：（可选）加入 Neo4j 知识图谱

处理 DrugBank 未直接记录的**间接相互作用**，通过 CYP450 酶路径推理：

```
例：华法林 + 青霉素
DrugBank 无直接相互作用记录
但：华法林 → 由 CYP2C9 代谢
    青霉素 → 抑制 CYP2C9
→ 图谱推理：存在间接相互作用风险，华法林血药浓度可能升高
```

---

## 注意事项

### .gitignore 说明

以下文件不会被上传到 GitHub：

```
drug_db.xml          # 原始数据，1.77GB
drug_db.xml.zip      # 压缩版，174MB （微信）
drug_structured.db   # SQLite 数据库，580MB
chroma_db/           # 向量数据库目录
chat_history/        # 对话历史
md5.text             # 去重记录
.env                 # API Key，根据环境变量去配置
```

### 技术栈

| 组件         | 选型                            |
| ------------ | ------------------------------- |
| LLM          | GPT-4o-mini (OpenAI)            |
| Embedding    | text-embedding-3-large (OpenAI) |
| 向量数据库   | ChromaDB                        |
| 结构化数据库 | SQLite                          |
| RAG 框架     | LangChain                       |
| 前端界面     | Streamlit                       |
| 数据来源     | DrugBank XML                    |

---
