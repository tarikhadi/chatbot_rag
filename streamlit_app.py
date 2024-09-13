import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

PDF_PATH = "bloomberggpt.pdf"

# Load PDF and split into smart chunks
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return pages

def create_smart_chunks(pages, chunk_size=800, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

def store_in_vector_db(chunks):
    embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')  # Correct model name
    vectorstore = Chroma.from_documents(chunks, embedding=embedding_model)
    return vectorstore

def process_pdf_for_rag(pdf_path):
    pages = load_pdf(pdf_path)
    chunks = create_smart_chunks(pages)
    vectorstore = store_in_vector_db(chunks)
    return vectorstore

st.title("RAG Chatbot - BloombergGPT PDF")

@st.cache_resource
def get_vectorstore():
    return process_pdf_for_rag(PDF_PATH)

vectorstore = get_vectorstore()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user prompt
if prompt := st.chat_input("What would you like to know about BloombergGPT?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using the instructions and chat history
    llm = ChatOpenAI(model="gpt-4")  # Use a valid model name
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
        return_source_documents=True,
    )

    with st.chat_message("assistant"):
        response = qa_chain({"query": prompt})
        result = response['result']
        st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})
