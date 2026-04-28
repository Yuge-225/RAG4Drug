import json
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from rag import RagService
import data_configuration as config

# ── Drugs available in the knowledge graph ────────────────────────────────────
GRAPH_DRUGS = [
    "Warfarin", "Aspirin", "Ibuprofen", "Simvastatin", "Amlodipine",
    "Metformin", "Clopidogrel", "Omeprazole", "Digoxin", "Fluconazole",
    "Amiodarone", "Phenytoin", "Rifampin", "Atorvastatin", "Lisinopril",
]

_GRAPH_HTML_PATH = Path(__file__).parent / "drug_interaction_graph.html"

# Injected into the iframe: hide standalone chrome, reveal legend, soften bg
_EMBED_OVERRIDE = """
<style>
  header      { display: none !important; }
  .left-panel { display: none !important; }
  body        { overflow: hidden; background: #0f172a; }
  .main       { padding: 0 !important; }
  #legend-overlay { display: block !important; }
</style>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_raw_html() -> str:
    return _GRAPH_HTML_PATH.read_text(encoding="utf-8")


def build_embedded_graph(highlighted: list[str]) -> str:
    html = _load_raw_html()
    drugs_js = json.dumps(highlighted)
    html = html.replace("var HIGHLIGHT_DRUGS = [];", f"var HIGHLIGHT_DRUGS = {drugs_js};")
    html = html.replace("</head>", f"{_EMBED_OVERRIDE}</head>")
    return html


def detect_graph_drugs(text: str) -> list[str]:
    return [
        d for d in GRAPH_DRUGS
        if re.search(rf"\b{re.escape(d)}\b", text, flags=re.IGNORECASE)
    ]


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG4Drug",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stHeader"] { display: none; }

  /* Comfortable reading width */
  .block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 860px !important;
  }

  /* Chat input bar */
  [data-testid="stBottom"] {
    background: #FAF9F6 !important;
    border-top: 1px solid #E8DDD6 !important;
  }

  /* No border on scrollable containers */
  [data-testid="stVerticalBlockBorderWrapper"] { border: none !important; }

  /* Graph iframe: rounded card + depth shadow */
  iframe {
    border: none !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 28px rgba(0,0,0,0.15) !important;
    display: block !important;
  }

  /* Tighten heading sizes inside chat responses */
  [data-testid="stChatMessageContent"] h1 { font-size: 1.1rem  !important; margin-top: 8px  !important; }
  [data-testid="stChatMessageContent"] h2 { font-size: 1.0rem  !important; margin-top: 6px  !important; }
  [data-testid="stChatMessageContent"] h3 { font-size: 0.93rem !important; }
  [data-testid="stChatMessageContent"] p  { font-size: 0.91rem !important; line-height: 1.7 !important; }

  /* Expander: blend into page */
  [data-testid="stExpander"] {
    border: 1px solid #E8DDD6 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #FAF9F6 !important;
  }
  [data-testid="stExpanderToggleIcon"] { color: #C96442 !important; }
</style>
""", unsafe_allow_html=True)

# ── Centered header ───────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: #FAF9F6;
  border-bottom: 1px solid #E8DDD6;
  padding: 14px 32px 13px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 13px;
  margin: 0 -2rem 0 -2rem;
">
  <div style="
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #C96442 0%, #E8966A 100%);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; color: #fff; flex-shrink: 0;
  ">⬡</div>
  <div>
    <div style="font-size:16px; font-weight:700; color:#1E1A16; letter-spacing:0.02em; line-height:1.2;">
      RAG4Drug
    </div>
    <div style="font-size:10px; color:#9B8579; font-family:monospace; letter-spacing:0.12em;">
      DRUG INTERACTION INTELLIGENCE
    </div>
  </div>
</div>
<div style="height:16px"></div>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "message" not in st.session_state:
    st.session_state["message"] = [{
        "role": "assistant",
        "content": (
            "Hi! I'm your drug interaction analyst powered by **RAG4Drug**.\n\n"
            "Ask me about any drug combination, for example:\n"
            "> *\"Can warfarin and aspirin be taken together?\"*\n\n"
            "When recognized drugs are mentioned, an **interaction graph** will appear "
            "below the response automatically."
        ),
    }]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

if "highlighted_drugs" not in st.session_state:
    st.session_state["highlighted_drugs"] = []

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state["message"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Interaction graph (shown below conversation when drugs are detected) ───────
highlighted = st.session_state["highlighted_drugs"]
if highlighted:
    drug_label = "  ·  ".join(highlighted)
    with st.expander(f"🔗  Interaction Graph  —  {drug_label}", expanded=True):
        components.html(
            build_embedded_graph(highlighted),
            height=500,
            scrolling=False,
        )

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask about drug interactions…")

if prompt:
    st.session_state["message"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    ai_res_list: list[str] = []
    res_stream = st.session_state["rag"].chain.stream(
        {"question": prompt}, config.session_config
    )

    def capture(generator, cache):
        for chunk in generator:
            cache.append(chunk)
            yield chunk

    with st.chat_message("assistant"):
        st.write_stream(capture(res_stream, ai_res_list))

    st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})

    detected = detect_graph_drugs(prompt)
    if detected != st.session_state["highlighted_drugs"]:
        st.session_state["highlighted_drugs"] = detected
        st.rerun()