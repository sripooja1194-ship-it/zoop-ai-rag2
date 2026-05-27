import streamlit as st
import time
import os
import tempfile
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image
from groq import Groq
import whisper
from streamlit_mic_recorder import mic_recorder

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Zoop AI RAG", layout="wide")

# ---------------- GROQ CLIENT ---------------- #
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ API Key missing! Please set it in Streamlit secrets.")
    st.stop()

client = Groq(api_key=api_key)

# ---------------- SESSION STATE ---------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ---------------- EMBEDDINGS & MODELS (CACHED) ---------------- #
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

embeddings = get_embeddings()

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# ---------------- PDF PROCESSING ---------------- #
def process_pdfs(uploaded_files):
    # Slightly larger chunks means more information fits into fewer pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=150
    )

    all_texts = []
    all_metadatas = []

    for file in uploaded_files:
        if file.name in st.session_state.processed_files:
            continue

        pdf_reader = PdfReader(file)
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        chunks = splitter.split_text(text)

        for chunk in chunks:
            all_texts.append(f"[Document Source: {file.name}]\n{chunk}")
            all_metadatas.append({"source": file.name})

        st.session_state.processed_files.add(file.name)

    if not all_texts:
        return st.session_state.vectorstore

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_texts(
            all_texts,
            embeddings,
            metadatas=all_metadatas
        )
    else:
        st.session_state.vectorstore.add_texts(
            all_texts,
            metadatas=all_metadatas
        )

    return st.session_state.vectorstore

# ---------------- LLM FUNCTION ---------------- #
def ask_llm(prompt, model_choice, temp, max_tok):
    response = client.chat.completions.create(
        model=model_choice,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=max_tok
    )
    return response.choices[0].message.content

# ---------------- RAG CONTEXT PIPELINE ---------------- #
def get_context(question):
    if not st.session_state.vectorstore:
        return ""

    # TUNED DOWN: Reduced total matched chunks to stay under Groq's 6K Free Tier Limit
    docs = st.session_state.vectorstore.similarity_search(question, k=15)

    grouped = {}
    for d in docs:
        source = d.metadata.get("source", "Unknown PDF")
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(d.page_content)

    final_context = ""
    for source, chunks in grouped.items():
        final_context += f"\n\n=========================================\n"
        final_context += f"📄 FILE IDENTIFIER: {source}\n"
        final_context += f"=========================================\n\n"
        # Only taking the top 4 highly relevant chunks per file to protect the rate limits
        final_context += "\n\n---\n\n".join(chunks[:4])

    return final_context

# ---------------- WELCOME SCREEN ---------------- #
if len(st.session_state.chat_history) == 0:

    st.markdown("""
    # 👋 Welcome to Zoop AI

    Upload multiple PDFs and cleanly chat/summarize across documents without blend errors.

    ### 📄 Features

    ✅ PDF Chat

    ✅ AI Answers

    ✅ Semantic Search

    ✅ Fast Responses

    
    Upload documents from the sidebar 👈
    """)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:

    st.title("🤖 Zoop AI Assistant")

    st.info("🚀 True AI Assistant")

    st.success("✅ Multiple PDF Support")

    st.success("✅ RAG Enabled")

    st.success("✅ PHI3 Mini Local AI Model")

    st.success("✅ Chat Memory")

    st.success("✅ Voice Enabled")

    st.success("✅ Image AI Support")

    st.markdown("---")

    if st.button("➕ New Chat"):
        st.session_state.chat_history = []
        st.session_state.last_query = ""
        st.rerun()

    # Upload PDFs
    st.subheader("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            vs = process_pdfs(uploaded_files)
            if vs is not None:
                st.session_state.vectorstore = vs
                st.success(f"Tracking: {len(st.session_state.processed_files)} file(s)")
            else:
                st.error("PDF could not be processed")

    st.markdown("---")

    # Upload Image
    st.subheader("🖼 Upload Image")
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("---")

    # AI Model Selection & Hyperparameters
    st.subheader("🧠 AI Model")
    selected_model = st.selectbox("Choose Model", ["llama-3.1-8b-instant", "llama3-70b-8192"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
    
    # Cap maximum return tokens to save room for prompt context
    max_tokens = st.slider("Max Response Space Tokens", 300, 1200, 600)

    st.markdown("---")

    # Quick Actions
    st.subheader("⚡ Quick Actions")
    click_query = ""
    if st.button("📋 Summarize PDFs Separately"):
        click_query = "Please perform an isolated, brief summary for each individual document found in the context. Split them cleanly with markdown titles."
    if st.button("🧠 Key Points"):
        click_query = "Provide the key bullet points from each document separately, stating the document source name first."
    if st.button("📘 Explain Simply"):
        click_query = "Explain the concepts in this document simply as if I am 10 years old."
    if st.button("❓ Generate Questions"):
        click_query = "Generate 5 practice questions based on this document context."
    st.markdown("---")

    # Download Chat
    chat_text = "".join([f"{msg['role']}: {msg['content']}\n\n" for msg in st.session_state.chat_history])
    st.download_button("💾 Download Chat", chat_text, file_name="zoop_chat.txt")

    # Clear Chat
    if st.button("🧹 Clear Workspace"):
        st.session_state.chat_history = []
        st.session_state.processed_files = set()
        st.session_state.vectorstore = None
        st.session_state.last_query = ""
        st.rerun()

# ---------------- CHAT DISPLAY ---------------- #
for msg in st.session_state.chat_history:
    avatar = "🤖" if msg["role"] == "assistant" else "👨"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------- INPUT CAPTURE ---------------- #
text_query = st.chat_input("Ask something from your PDF...")
if click_query:
    text_query = click_query

audio = mic_recorder()
if audio and "bytes" in audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio["bytes"])
        tmp_path = tmp.name 
    try:
        if os.path.exists(tmp_path):
            result = whisper_model.transcribe(tmp_path)
            text_query = result["text"]
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------- MAIN CHAT FLOW ---------------- #
if text_query and text_query != st.session_state.last_query:
    st.session_state.last_query = text_query

    st.session_state.chat_history.append({"role": "user", "content": text_query})
    with st.chat_message("user", avatar="👨"):
        st.markdown(text_query)

    context = get_context(text_query)

    if not context.strip():
        prompt = f"The user asked: '{text_query}'. Respond politely asking them to upload a PDF file through the sidebar workspace first."
    else:
        prompt = f"""
You are a context-isolated Multi-PDF Assistant. Keep answers direct and concise to respect strict output token parameters.

EXPLICIT INSTRUCTIONS:
- Identify every distinct 'FILE IDENTIFIER' listed in the text below.
- Create a distinct Markdown section header for each individual document (e.g. ### 📄 Document: [Filename]).
- Provide a summary or specific answer targeting that file exclusively underneath its respective header.
- Do NOT merge text insights across different files. 

CONTEXT BLOCK:
{context}

USER DIRECTIVE:
{text_query}

CLEANLY SEPARATED INDEPENDENT ANSWER:
"""
        
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing document architectures..."):
            try:
                answer = ask_llm(prompt, selected_model, temperature, max_tokens)
                st.markdown(answer)
            except Exception as e:
                answer = f"Error communicating with AI engine: {str(e)}"
                st.error(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
