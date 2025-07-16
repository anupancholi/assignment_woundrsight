import streamlit as st
from src.retriever import ChunkRetriever
from src.generator import format_prompt, ollama_generate_stream
import numpy as np

# We firstly load retriever at app start


@st.cache_resource
def get_retriever():
    retriever = ChunkRetriever(
        vector_db_path="vectordb/faiss.index",
        meta_path="vectordb/chunk_metadata.npy"
    )
    return retriever


retriever = get_retriever()

st.title("📄 RAG Chatbot Demo")
st.sidebar.info(
    f"Model: Mistral (Ollama)\nChunks in DB: {len(retriever.chunk_texts)}")
if st.sidebar.button("Reset chat"):
    st.session_state.clear()

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.submitted = False

st.write("Ask anything about the document:")

query = st.text_input("Your question:", key="query")

if st.button("Submit") or (query and st.session_state.get("submitted", False)):
    st.session_state.submitted = True
    results = retriever.retrieve(query, top_k=4)
    st.write("**Retrieved Chunks:**")
    for r in results:
        st.write(
            f"**Chunk {r['idx']} (Score: {r['score']:.2f}):** {r['chunk'][:300]}...")
    prompt = format_prompt(results, query)
    st.write("**AI Response:**")
    answer = ""
    with st.spinner("Generating..."):
        placeholder = st.empty()
        for chunk in ollama_generate_stream(prompt, model="mistral"):
            answer += chunk
            placeholder.markdown(f"{answer}▌")
    st.session_state.history.append({"question": query, "answer": answer})

if st.session_state.history:
    st.write("## Chat History")
    for turn in st.session_state.history:
        st.markdown(
            f"**Question:** {turn['question']}\n\n**Answer:** {turn['answer']}")
