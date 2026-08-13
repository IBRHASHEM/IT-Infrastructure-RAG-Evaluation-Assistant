import streamlit as st
from config import MODEL_NAME, EMBEDDING_MODEL
from vector_store import VectorStore
from rag import RAG
import os
pdf_count = len([
    f for f in os.listdir("data")
    if f.endswith(".pdf")
])
st.set_page_config(
    page_title="IT Infrastructure Assistant",
    page_icon="💻",
    layout="wide"
)

assistant = RAG()
vector_db = VectorStore()

#=======================
# ======================================
# Sidebar
# ======================================

with st.sidebar:

    st.title("💻 IT Infrastructure Assistant\nEnterprise RAG System")

    st.divider()

    st.subheader("🤖 Model")
    st.code(MODEL_NAME)

    st.subheader("🧠 Embeddings")
    st.caption(EMBEDDING_MODEL)

    st.divider()

    try:

        collection = vector_db.collection

        chunks = collection.count()

    except:

        chunks = "Unknown"

    st.metric(
        "🧩 Indexed Chunks",
        chunks
    )

    st.metric(
        "📄 Documents",
        pdf_count

    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption("IT Infrastructure Assistant")

    st.caption("RAG + ChromaDB + Gemini")
# -------------------------
# Chat History
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💻 IT Infrastructure Assistant")
st.markdown("Ask questions about your indexed IT documentation.")

# -------------------------
# Display previous messages
# -------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:

            with st.expander("Sources"):

                for s in message["sources"]:

                    st.write(
                        f"📄 {s['source']} (Page {s['page']})"
                    )

# -------------------------
# User Input
# -------------------------
question = st.chat_input("Ask your question...")

if question:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant
    with st.chat_message("assistant"):

        with st.spinner("Searching documentation..."):

            result = assistant.ask(question)

        st.markdown(result["answer"])

        with st.expander("Sources"):

            for s in result["sources"]:

                st.write(
                    f"📄 {s['source']} (Page {s['page']})"
                )

    # Save history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        }
    )