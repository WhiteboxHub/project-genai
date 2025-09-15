from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import os


def load_documents():
    """
    Example: Load raw text documents.
    Replace this with your actual document loading logic.
    """
    texts = [
        "Machine learning enables computers to learn from data.",
        "Natural Language Processing is a subfield of AI focused on text and speech.",
        "Vector databases store embeddings for efficient retrieval."
    ]
    docs = [Document(page_content=t) for t in texts]
    return docs


def create_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize Hugging Face embeddings.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def split_documents(documents, chunk_size: int = 300, chunk_overlap: int = 50):
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def build_index(documents, embeddings, index_path: str = "faiss_index"):
    """
    Build FAISS vector index and save it locally.
    """
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save FAISS index
    vectorstore.save_local(index_path)
    print(f"✅ Index saved at: {index_path}")
    return vectorstore


if __name__ == "__main__":
    # Step 1: Load documents
    docs = load_documents()

    # Step 2: Create embeddings
    embeddings = create_embeddings()

    # Step 3: Split documents
    chunks = split_documents(docs)

    # Step 4: Build index
    build_index(chunks, embeddings)
