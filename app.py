import streamlit as st
import time
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
# Safely pulling from st.secrets as defined later in your code
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    # Fallback to hardcoded if secrets aren't set up yet
    client = Groq(api_key="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") 

# ---------------- SESSION STATE ---------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "full_pdf_text" not in st.session_state:  # <-- CRITICAL FIX: To hold raw text for global summaries
    st.session_state.full_pdf_text = ""

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ---------------- WELCOME SCREEN ---------------- #

if len(st.session_state.chat_history) == 0:

    st.markdown("""
    # 👋 Welcome to Zoop AI

    Upload PDFs and chat with your documents using AI.

    ### 📄 Features

    ✅ PDF Chat

    ✅ AI Answers

    ✅ Semantic Search

    ✅ Fast Responses

    ---
    
    Upload documents from the sidebar 👈
    """)

# ---------------- EMBEDDINGS & MODELS (CACHED) ---------------- #
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# ---------------- PDF PROCESSING ---------------- #
def process_pdfs(uploaded_files):
    text = ""
    for file in uploaded_files:
        if file.name in st.session_state.processed_files:
            continue
        
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        st.session_state.processed_files.add(file.name)

    if not text.strip():
        return None

    # Keep track of global text for summarization tasks
    st.session_state.full_pdf_text += "\n" + text

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    if not chunks:
        return None

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ---------------- LLM FUNCTION ---------------- #
def ask_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature if 'temperature' in locals() else 0.2,
        max_tokens=max_tokens if 'max_tokens' in locals() else 300
    )
    return response.choices[0].message.content

# ---------------- RAG CONTEXT PIPELINE ---------------- #
def get_context(question):
    # CRITICAL FIX: If the user explicitly asks for a summary/entire overview, bypass similarity search
    summary_keywords = ["summarize", "summary", "give me an overview", "explain the whole", "key points"]
    if any(keyword in question.lower() for keyword in summary_keywords):
        if st.session_state.full_pdf_text:
            # Return a trimmed version if it's exceptionally long to respect model context window
            return st.session_state.full_pdf_text[:15000] 
        
    if not st.session_state.vectorstore:
        return ""

    docs = st.session_state.vectorstore.similarity_search(question, k=4)
    return "\n".join([d.page_content for d in docs])

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

    # New Chat
    if st.button("➕ New Chat"):

        for msg in st.session_state.chat_history:
         st.chat_message(...)


    # Upload PDFs
    st.subheader("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            vs = process_pdfs(uploaded_files)
            if vs is not None:
                st.session_state.vectorstore = vs
                st.success("PDF processed successfully")
            elif st.session_state.vectorstore is None:
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
    selected_model = st.selectbox("Choose Model", ["llama-3.1-8b-instant", "mistral", "phi3:mini"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.slider("Max Tokens", 100, 2000, 500)

    st.markdown("---")

    # Quick Actions (Intercepted and wired into the chat framework)
    st.subheader("⚡ Quick Actions")
    click_query = ""
    if st.button("🧠 Key Points"):
        click_query = "Provide the key bullet points from the document."
    if st.button("📘 Explain Simply"):
        click_query = "Explain the concepts in this document simply as if I am 10 years old."
    if st.button("❓ Generate Questions"):
        click_query = "Generate 5 practice questions based on this document context."

        st.markdown("---")

    # Download Chat
    chat_text = ""

    for msg in st.session_state.chat_history:

        chat_text += f"{msg['role']}: {msg['content']}\n\n"

    st.download_button(
        "💾 Download Chat",
        chat_text,
        file_name="zoop_chat.txt"
    )

    st.markdown("---")

    # Clear Chat
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.processed_files = set()
        st.session_state.full_pdf_text = ""
        st.session_state.vectorstore = None
        st.rerun()


# ---------------- CHAT DISPLAY ---------------- #
for msg in st.session_state.chat_history:
    avatar = "🤖" if msg["role"] == "assistant" else "👨"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------- INPUT CAPTURE ---------------- #
text_query = st.chat_input("Ask something from your PDF...")

# Voice Input Handling
audio = mic_recorder(start_prompt="🎤 Voice Input", stop_prompt="⏹ Stop Recording", just_once=True, use_container_width=True)
if audio:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio["bytes"])
    result = whisper_model.transcribe("temp_audio.wav")
    text_query = result["text"]

# Overwrite query if a Quick Action button was clicked
if click_query:
    text_query = click_query

# ---------------- MAIN CHAT FLOW Execution ---------------- #
if text_query and text_query != st.session_state.last_query:
    st.session_state.last_query = text_query

    # Append and show User Message
    st.session_state.chat_history.append({"role": "user", "content": text_query})
    with st.chat_message("user", avatar="👨"):
        st.markdown(text_query)

    # Context retrieval with upgraded smart checks
    context = get_context(text_query)

    if not context.strip():
        prompt = f"The user is asking: '{text_query}'. Politely inform them to upload a PDF context first to get started."
    else:
        prompt = f"""
You are Zoop AI assistant.

You can understand:
- Hindi
- English
- Hinglish

If user asks in Hindi, answer in Hindi.
If user asks in English, answer in English.
If user asks in Hinglish, answer naturally in Hinglish.

Answer using the provided context.
If it is a summary request, construct a comprehensive synthesis across all details given.

Context:
{context}

Question:
{text_query}
"""

    # Run LLM
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                answer = ask_llm(prompt)
                st.markdown(answer)
            except Exception as e:
                answer = f"Error: {str(e)}"
                st.error(answer)

    # Save Assistant Response
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
