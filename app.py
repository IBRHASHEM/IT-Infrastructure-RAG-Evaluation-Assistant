import os
import time

import streamlit as st

from config import MODEL_NAME, EMBEDDING_MODEL_PATH
from vector_store import VectorStore
from rag_hybrid import HybridRAG


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="IT Servers Support Assistant",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DOCUMENT COUNT
# =========================================================

pdf_count = len(
    [
        f
        for f in os.listdir("data")
        if f.lower().endswith(".pdf")
    ]
)


# =========================================================
# LOAD RAG
# =========================================================

@st.cache_resource
def load_rag():
    return HybridRAG()


@st.cache_resource
def load_vector_store():
    return VectorStore()


assistant = load_rag()
vector_db = load_vector_store()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("💻 IT Servers Support Assistant")

    st.caption("Enterprise RAG System")

    st.divider()

    # -----------------------------------------------------
    # Generation Model
    # -----------------------------------------------------

    st.subheader("🤖 Generation Model")

    st.code(
        MODEL_NAME,
        language="text"
    )

    # -----------------------------------------------------
    # Embedding Model
    # -----------------------------------------------------

    st.subheader("🧠 Embedding Model")

    st.code(
        EMBEDDING_MODEL_PATH,
        language="text"
    )

    st.divider()

    # -----------------------------------------------------
    # Knowledge Base Statistics
    # -----------------------------------------------------

    st.subheader("📊 Knowledge Base")

    try:
        chunks = vector_db.collection.count()
    except Exception:
        chunks = "Unknown"

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🧩 Chunks",
            chunks
        )

    with col2:
        st.metric(
            "📄 PDFs",
            pdf_count
        )

    st.divider()

    # -----------------------------------------------------
    # Retrieval Architecture
    # -----------------------------------------------------

    st.subheader("🔎 Retrieval")

    st.caption(
        "Vector Search + BM25 + RRF"
    )

    st.divider()

    # -----------------------------------------------------
    # System Status
    # -----------------------------------------------------

    st.subheader("🟢 System")

    st.caption("RAG: Ready")
    st.caption("Vector DB: ChromaDB")
    st.caption("Embeddings: Local BGE")
    st.caption("Generator: Local Qwen")

    st.divider()

    # -----------------------------------------------------
    # Clear Chat
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "Local BGE + ChromaDB + BM25 + RRF + Local Qwen"
    )


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# MAIN PAGE
# =========================================================

st.title(
    "💻 IT Servers Support Assistant"
)

st.markdown(
    "Ask questions about your indexed IT infrastructure documentation."
)


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Sources"):

                for source in message["sources"]:

                    st.write(
                        f"📄 {source['source']} "
                        f"(Page {source['page']})"
                    )

        # -------------------------------------------------
        # Performance Information
        # -------------------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("elapsed_time") is not None
        ):

            st.caption(
                f"⏱️ Response time: "
                f"{message['elapsed_time']:.2f} seconds"
            )


# =========================================================
# USER INPUT
# =========================================================

question = st.chat_input(
    "Ask your question..."
)


if question:

    # -----------------------------------------------------
    # User message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        start_time = time.perf_counter()

        with st.spinner(
            "Searching documentation and generating answer..."
        ):

            try:

                result = assistant.ask(
                    question
                )

                elapsed_time = (
                    time.perf_counter()
                    - start_time
                )

                answer = result.get(
                    "answer",
                    "I don't know based on the indexed documentation."
                )

                sources = result.get(
                    "sources",
                    []
                )

                st.markdown(answer)

                # -------------------------------------------------
                # Sources
                # -------------------------------------------------

                if sources:

                    with st.expander(
                        "📚 Sources"
                    ):

                        for source in sources:

                            st.write(
                                f"📄 {source['source']} "
                                f"(Page {source['page']})"
                            )

                # -------------------------------------------------
                # Performance
                # -------------------------------------------------

                st.caption(
                    f"⏱️ Response time: "
                    f"{elapsed_time:.2f} seconds"
                )

            except Exception as e:

                elapsed_time = (
                    time.perf_counter()
                    - start_time
                )

                answer = (
                    "An error occurred while processing "
                    "your question."
                )

                sources = []

                st.error(
                    f"Error: {e}"
                )

    # -----------------------------------------------------
    # Save assistant message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "elapsed_time": elapsed_time,
        }
    )