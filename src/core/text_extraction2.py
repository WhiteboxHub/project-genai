from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader,DirectoryLoader, PyPDFLoader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pinecone import PineconeException
from generation import generate_response
from langchain_groq import ChatGroq
from chunking import file_chunking

from embeding import Embed_model
from langchain.schema import Document
from typing import List
import os

from dotenv import load_dotenv
from storing import Pineconedb

load_dotenv()
folder_path = os.getenv("pdf_folder_path")
#all_docs = load_pdfs_from_directory(folder_path)

def load_pdf_with_langchain(filepath):
    loader = PyMuPDFLoader(filepath)
    documents = loader.load()
    return documents

'''res=load_pdf_with_langchain(r"Data\\Medical_book.pdf")
print(res[1])'''

def load_csv_with_langchain(filepath):
    loader=CSVLoader(filepath)
    document=loader.load()
    return document

'''res2=load_csv_with_langchain(r"Data\\creditcard.csv")
for doc in res2[:3]:
    print(doc.page_content)'''

def load_pdfs_from_directory(directory_path: str):
    loader = DirectoryLoader(
        path=directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


if __name__ == "__main__":
    # Load PDF docs
    
    all_docs = load_pdfs_from_directory(folder_path)
    print(f"Loaded {len(all_docs)} documents.")

    '''
    for i, doc in enumerate(all_docs[:3]):  # Preview first 3 docs
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content[:500])  # Show first 500 characters'''
    
    all_chunks = []
    for doc in all_docs:
     chunks = file_chunking.recursive(doc.page_content, chunk_size=500, overlap=50)
     all_chunks.extend(chunks)

 # Preview the first chunk
    if all_chunks:
        print(f"\n Total Chunks Created: {len(all_chunks)}")
        print(f"\n First Chunk Preview:\n{all_chunks[0]}")
    else:
        print("No chunks created.")

    embeddings1=Embed_model.sentence_Transfoer(all_chunks, model_name="all-MiniLM-L6-v2")
    print(embeddings1[1])

    '''create_store=Pineconedb.create_index("gen-ai", 384, metric="cosine", cloud="aws", region="us-east-1")

    storing=Pineconedb.store_embeddings_pinecone("gen-ai", all_chunks, embeddings1)'''
    query="Explain what is Agentic AI?"
    retrieved_data=Pineconedb.retrieve_data_from_pinecone("gen-ai", query=query, top_k=5, model_name="all-MiniLM-L6-v2")
    print(retrieved_data)
    context_str = "\n".join([doc.page_content for doc in retrieved_data])
    final_result=generate_response(context_str, query)
    print(final_result)

    #print(f"\nRetrieved context preview:\n{context_str[:500]}")
    
    '''vector_store = ChromaDB.store_data(all_chunks)
    print(vector_store._collection.count())

    # Prepare embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Run a test query
    query = "Explain what is langchain?"
    retrieved_text= ChromaDB.retrieve_data(query, model=embedding_model, collection=vector_store)
    context_str = "\n".join([doc.page_content for doc in retrieved_text])'

    #print(f"\nRetrieved context preview:\n{context_str[:500]}")

    final_result=generate_response(context_str, query)
    print(final_result)
    '''
    
