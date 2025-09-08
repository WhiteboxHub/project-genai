from src.core.loading import file_load
from src.core.chunking import file_chunking
from src.core.embeding import embed_model
from src.core.storing import Milvus
from dotenv import load_dotenv
from src.core.generation import LLMGenerator
import os


def load_data(milvus_obj: Milvus):
    text_from_file = file_load.read_text_from_file("Data/Gutenburg.txt")
    chunked_file = file_chunking.recursive(text_from_file)
    print("****")
    print(f"Recursive chunking created {len(chunked_file)} chunks")
    embeddings = embed_model.sentence_transformer(chunked_file)
    milvus_vector_store = milvus_obj.store_data(chunked_file, embeddings)


def retrive_top_docs(milvus_obj: Milvus, query):
    embeddings = embed_model.sentence_transformer([query])
    embedding_to_query = embeddings[0]
    k_docs = milvus_obj.retrive_data(embedding_to_query)
    return k_docs

def generate_answer(k_docs, query):
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    obj = LLMGenerator(groq_api_key)
    response = obj.generate_response_from_documents(k_docs, query)
    return response


if __name__ == "__main__":
    milvus_obj = Milvus()
    load_data(milvus_obj)
    query = "What is the rabbit hole?"
    k_docs = retrive_top_docs(milvus_obj, query)
    response = generate_answer(k_docs, query)
    print(response)

    

    

