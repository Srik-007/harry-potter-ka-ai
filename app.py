import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)
prompt = ChatPromptTemplate.from_template(
    """
    answer the question based on the provided context only. please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question:{input}
    """
)
import re

def clean_response(text):
    # Remove anything between <think>...</think>
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def create_vectors():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = PyPDFDirectoryLoader("jk rowling")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.documents[0:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_document, st.session_state.embeddings)

# UI
user_prompt = st.text_input("Enter your query")

if st.button("Document embeddings"):
    create_vectors()
    st.success("Vectorbase is ready")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please create the document embeddings first by clicking the button above.")
    else:
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed = time.process_time() - start
        st.write(f"Response time: {elapsed:.2f} seconds")
        st.write(clean_response(response["answer"]))


        with st.expander("Document similarity search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("----------xxx--------------")
