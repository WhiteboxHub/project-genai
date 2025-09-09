from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader,DirectoryLoader, PyPDFLoader
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from generation import generate_response
from langchain_groq import ChatGroq
from chunking import file_chunking
from langchain.schema import Document
from typing import List
import os
from dotenv import load_dotenv
from storing import ChromaDB

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

    vector_store = ChromaDB.store_data(all_chunks)

    # Prepare embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Run a test query
    query = "Explain what is langchain?"
    retrieved_text= ChromaDB.retrieve_data(query, model=embedding_model, collection=vector_store)
    context_str = "\n".join([doc.page_content for doc in retrieved_text])

    #print(f"\nRetrieved context preview:\n{context_str[:500]}")

    final_result=generate_response(context_str, query)
    print(final_result)
    
    
