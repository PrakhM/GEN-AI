import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            #print("page text:",text);
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], 
        chunk_size = 1000, chunk_overlap = 200, length_function = len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore
    

def main():
    
    load_dotenv()
    st.set_page_config(
        page_title="Document Q&A",
        page_icon="📄",
        layout="wide"
    )

    col1, col2, col3 = st.columns([1, 2.2, 1])

    with col2:
        st.text_input("Ask", placeholder="Type your question here...", label_visibility="collapsed")

    with st.sidebar:
        st.subheader("Documents")

        docs = st.file_uploader("Upload", label_visibility="collapsed", accept_multiple_files = True)
        if st.button("Process"):
            if docs:
                with st.spinner("Processing..."):
                        raw_text = get_pdf_text(docs)
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vectorstore(text_chunks)
            else:
                st.warning("Please upload at least one PDF.")

if __name__ == '__main__':
    main()