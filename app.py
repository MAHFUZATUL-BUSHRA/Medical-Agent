"""Streamlit web UI for the multi-tool medical agent.

    streamlit run app.py
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from medagent import config
from medagent.schemas import CANCER, HEART

st.set_page_config(page_title="Medical Multi-Tool Agent", page_icon="🩺", layout="wide")

TOOL_BADGES = {
    "HeartDiseaseDBTool": "❤️ HeartDiseaseDBTool",
    "CancerDBTool": "🎗️ CancerDBTool",
    "MedicalWebSearchTool": "🌐 MedicalWebSearchTool",
}

EXAMPLES = {
    "❤️ Heart data": [
        "How many patients in the heart dataset have heart disease?",
        "What is the average cholesterol for males vs females with heart disease?",
        "How many patients are older than 60?",
    ],
    "🎗️ Cancer data": [
        "What percentage of smokers have a cancer diagnosis?",
        "What is the average BMI of patients with high genetic risk?",
        "How many patients have a family history of cancer and a diagnosis?",
    ],
    "🌐 General knowledge": [
        "What are the symptoms of a heart attack?",
        "What is chemotherapy?",
        "What is hypertension, and what is the average resting blood pressure in the heart dataset?",
    ],
}


@st.cache_resource
def get_worker() -> ThreadPoolExecutor:
    """One long-lived thread with its own event loop.

    The OpenAI Agents SDK is async; running every call on the same thread/loop avoids
    'event loop is closed' errors when Streamlit re-runs the script.
    """
    return ThreadPoolExecutor(max_workers=1, initializer=lambda: asyncio.set_event_loop(asyncio.new_event_loop()))


def get_chat(backend: str):
    chats = st.session_state.setdefault("chats", {})
    if backend not in chats:
        def make():
            if backend == "langchain":
                from medagent.lc_agent import LangChainChat
                return LangChainChat()
            from medagent.agents_sdk import AgentsSDKChat
            return AgentsSDKChat()
        chats[backend] = get_worker().submit(make).result()
    return chats[backend]


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🩺 Settings")
    backend_label = st.radio(
        "Router backend",
        ["OpenAI Agents SDK", "LangChain AgentExecutor"],
        help="Both routers use the same three tools.",
    )
    backend = "langchain" if backend_label.startswith("LangChain") else "agents"

    st.divider()
    st.subheader("Status")
    st.write(f"**LLM:** `{config.LLM_PROVIDER}` / `{config.LLM_MODEL}`")
    st.write("LLM key: " + ("✅ set" if config.LLM_API_KEY else "❌ missing"))
    import os
    search_key = "SERPAPI_API_KEY" if config.SEARCH_PROVIDER == "serpapi" else "TAVILY_API_KEY"
    st.write(f"Search key ({config.SEARCH_PROVIDER}): " + ("✅ set" if os.getenv(search_key) else "❌ missing"))
    for spec in (HEART, CANCER):
        ok = (config.DB_DIR / spec.db_file).exists()
        st.write(f"`{spec.db_file}`: " + ("✅ found" if ok else "❌ run build_databases"))

    st.divider()
    st.subheader("Try an example")
    for group, questions in EXAMPLES.items():
        with st.expander(group):
            for i, q in enumerate(questions):
                if st.button(q, key=f"{group}{i}", use_container_width=True):
                    st.session_state["pending"] = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state.pop("chats", None)
        st.rerun()

# ---------------- main ----------------
st.title("🩺 Medical Multi-Tool Agent")
st.caption(
    "Ask data questions about the Heart Disease and Cancer datasets (SQL), "
    "or general medical questions (web search). Informational only, not medical advice."
)

messages = st.session_state.setdefault("messages", [])
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("tools"):
            st.caption("Tools used: " + " · ".join(TOOL_BADGES.get(t, t) for t in m["tools"]))

question = st.chat_input("Ask a question...") or st.session_state.pop("pending", None)

if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking... (the SQL agent may take a few seconds)"):
                chat = get_chat(backend)
                answer, tools = get_worker().submit(chat.ask, question).result()
            tools = list(dict.fromkeys(tools))  # de-duplicate, keep order
            st.markdown(answer)
            if tools:
                st.caption("Tools used: " + " · ".join(TOOL_BADGES.get(t, t) for t in tools))
            messages.append({"role": "assistant", "content": answer, "tools": tools})
        except Exception as exc:
            msg = f"⚠️ {type(exc).__name__}: {exc}"
            st.error(msg)
            messages.append({"role": "assistant", "content": msg, "tools": []})
