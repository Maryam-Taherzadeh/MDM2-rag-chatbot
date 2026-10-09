from pathlib import Path
import sys
import os

import streamlit as st


# ============================================================
# 1. Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. Load OpenAI API key
# ============================================================

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not configured.")
    st.stop()


# ============================================================
# 3. Import RAG pipeline
# ============================================================

from src.rag.rag_pipeline import rag_answer, candidate_df


# ============================================================
# 4. Page configuration
# ============================================================

st.set_page_config(
    page_title="MDM2-p53 Drug Discovery Assistant",
    page_icon="🧬",
    layout="wide"
)