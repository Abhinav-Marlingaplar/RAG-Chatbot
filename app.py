import streamlit as st
import time
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Streamlit App Title and Styling
st.markdown("<h1 style='text-align: center; color: #16324F;'>🤖 RAG-Based Chatbot using Gemini</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        .stApp {
            background-color: #D9F0FF;
        }
        section[data-testid="stSidebar"] {
            background-color: #A3D5FF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "data" not in st.session_state:
    st.session_state.data = []

# Sidebar for uploading PDFs
with st.sidebar:
    st.markdown("## 📂 Upload PDF")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Submit & Process"):
            with st.spinner("🔄 Processing your documents..."):
                all_data = []
                for file in uploaded_files:
                    temp_path = f"./temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())

                    loader = PyPDFLoader(temp_path)
                    all_data.extend(loader.load())

                    # Safe file cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                st.session_state.data = all_data
                st.success("✅ PDFs uploaded successfully! Now building embeddings...")

# Process documents if available
if "data" in st.session_state and st.session_state.data:
    data = st.session_state.data

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Use free local embeddings (no quota, no cost)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create or load Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Create retriever for similarity search
    st.session_state.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    st.success("✅ Document processed and vector store created!")

# Chat input section
query = st.chat_input("💬 Type your question here...")

if query and st.session_state.retriever:
    st.session_state.history.append({"user": query})

    # System prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n{context}"
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create QA chain
    question_answer_chain = create_stuff_documents_chain(
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0), 
        prompt
    )
    rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

    # Get model response
    response = rag_chain.invoke({"input": query})
    st.session_state.history.append({"assistant": response['answer']})

# Display chat history
if st.session_state.history:
    for chat in st.session_state.history:
        if "user" in chat:
            st.markdown(
                f"""
                <div style='background-color:#A3D5FF; padding:10px 15px; border-radius:10px;
                margin:10px 0; text-align:right; max-width: 80%; margin-left:auto;'>
                    <b>You:</b><br>{chat['user']}
                </div>
                """,
                unsafe_allow_html=True
            )
        if "assistant" in chat:
            st.markdown(
                f"""
                <div style='background-color:#83C9F4; padding:10px 15px; border-radius:10px;
                margin:10px 0; text-align:left; max-width: 80%; margin-right:auto;'>
                    <b>Assistant:</b><br>{chat['assistant']}
                </div>
                """,
                unsafe_allow_html=True
            )
