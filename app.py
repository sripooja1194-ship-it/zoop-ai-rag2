import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import whisper
import tempfile
import os
from streamlit_mic_recorder import mic_recorder

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Zoop AI RAG", layout="wide")

# ---------------- GROQ CLIENT ---------------- #
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ---------------- SESSION STATE ---------------- #

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:

    st.session_state.processed_files = set()

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

# ---------------- EMBEDDINGS (CACHED) ---------------- #
@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

embeddings = get_embeddings()

# ---------------- WHISPER MODEL ---------------- #
@st.cache_resource
def load_whisper():

    return whisper.load_model("base")

whisper_model = load_whisper()

# ---------------- PDF PROCESSING ---------------- #
def process_pdfs(uploaded_files):
 @st.cache_resource
 def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

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
                text += page_text

        st.session_state.processed_files.add(file.name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    if not chunks:
        return None

    vectorstore = FAISS.from_texts(
        chunks,
        embeddings
    )

    return vectorstore


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


    st.session_state.chat_history.append({
    "role": "assistant",
    "content": answer if 'answer' in locals() else "No response generated"
})
    st.markdown("---")

    # Upload PDFs
    st.subheader("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload your PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")

    # Upload Image
    st.subheader("🖼 Upload Image")

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    st.markdown("---")

    # AI Model
    st.subheader("🧠 AI Model")

    selected_model = st.selectbox(
        "Choose Model",
        [
            "phi3:mini",
            "llama3",
            "mistral"
        ]
    )

    st.markdown("---")

    # Settings
    st.subheader("⚙️ Settings")

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.2
    )

    max_tokens = st.slider(
        "Max Tokens",
        100,
        1000,
        300
    )

    st.markdown("---")

    # Quick Actions
    st.subheader("⚡ Quick Actions")

    key_points = st.button("🧠 Key Points")

    explain_simple = st.button("📘 Explain Simply")

    generate_questions = st.button("❓ Generate Questions")

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

    # Stats
    st.subheader("📊 Stats")

    st.metric(
        "Chats",
        len(st.session_state.chat_history)
    )

    if uploaded_files:

        st.metric(
            "Documents",
            len(uploaded_files)
        )

    st.markdown("---")

    # Clear Chat
    if st.button("🧹 Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

# ---------------- CHAT DISPLAY ---------------- #

for msg in st.session_state.chat_history:

    avatar = "🤖" if msg["role"] == "assistant" else "👨"

    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------- QUERY INPUT ---------------- #
# ---------------- VOICE INPUT ---------------- #

audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹ Stop Recording",
    just_once=True,
    use_container_width=True
)
voice_text = ""

if audio:

    with open("temp_audio.wav", "wb") as f:
        f.write(audio["bytes"])

    result = whisper_model.transcribe("temp_audio.wav")

    voice_text = result["text"]

    st.success(f"🎤 You said: {voice_text}")


# ---------------- RAG PIPELINE ---------------- #
def get_context(question):

    if not st.session_state.vectorstore:
        return ""

    docs = st.session_state.vectorstore.similarity_search(question, k=3)

    return "\n".join([d.page_content for d in docs])

def analyze_image(uploaded_image):

    def analyze_image_with_groq(image_bytes):
     response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": "Describe this image in detail.",
                "image": image_bytes
            }
        ]
    )

    return response.choices[0].message.content


from groq import Groq
import streamlit as st

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ---------------- MAIN CHAT FLOW (FIXED) ---------------- #

text_query = st.chat_input("Ask something from your PDF...")

# initialize memory
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if text_query and text_query != st.session_state.last_query:

    st.session_state.last_query = text_query

    # user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": text_query
    })

    with st.chat_message("user"):
        st.markdown(text_query)

    # context
    context = get_context(text_query)

    prompt = f"""
You are Zoop AI assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{text_query}
"""

    # run LLM safely
    try:
        answer = ask_llm(prompt, selected_model)
    except Exception as e:
        answer = f"Error: {str(e)}"

    # assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })