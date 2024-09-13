import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set API key
os.environ["OPENAI_API_KEY"] = ""

PDF_PATH = "/Users/tarikhadi/Desktop/untitled folder/bloomberggpt.pdf"


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
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small', disallowed_special=())
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

# Define instructions (not visible to the user, handled in the response generation logic)
INSTRUCTIONS = """
You are an AI assistant specialized in answering questions about the BloombergGPT article.
Your role is to provide accurate, concise, and helpful responses based on the information in the document.

Follow these guidelines:

1. If the answer is not contained in the context, politely state that you don't have enough information to answer accurately.
2. Provide direct and concise answers, but include relevant details when necessary.
3. If asked about technical details, explain them in a clear and understandable manner.
4. When discussing the capabilities or performance of BloombergGPT, be objective and avoid exaggeration.
5. Use appropriate terms and concepts related to AI, machine learning, and finance when relevant.
6. If the question is unclear, ask for clarification before attempting to answer.
7. Whenever user asks to see tables, ALWAYS show in a formatted structure.
8. Always end the response with a question suggesting the user to ask something related to BloombergGPT but which they have not asked yet.
"""

# Helper function to build conversation history into the prompt
def build_conversation_history(messages):
    history = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            history += f"User: {content}\n"
        else:
            history += f"Assistant: {content}\n"
    return history

# Handle user prompt
if prompt := st.chat_input("What would you like to know about BloombergGPT?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build full prompt, including instructions and conversation history
    conversation_history = build_conversation_history(st.session_state.messages)
    full_prompt = INSTRUCTIONS + "\n" + conversation_history + f"\nUser question: {prompt}"

    # Generate response
    llm = ChatOpenAI(model="gpt-4o-mini")  # Use the GPT model that supports answering
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
        return_source_documents=True,
    )

    # Generate response using the instructions, chat history, and user input
    with st.chat_message("assistant"):
        response = qa_chain({"query": full_prompt})

        # Extract the answer from the response
        result = response['result']
        
        st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})

# Suggest further questions
#st.write("Ask anything about the BloombergGPT article!")
