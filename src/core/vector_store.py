from langchain.vectorstores import Chroma
from typing import List, Optional, Sequence
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from core.embedding import embedding_model
from core.chunking import chunking
import numpy as np
import uuid


class VectorStore:

    @staticmethod
    def create_chroma_vector_store(embeddings, persist_directory="./chroma_db"):
        """
        Initializes and returns a Chroma vector store.
        """
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    @staticmethod
    def milvus_vector_store():
        # ---- CONFIG ----
        MILVUS_HOST = "127.0.0.1"
        MILVUS_PORT = "19530"
        COLLECTION = "docs"
        MODEL_NAME = "embedding_model"  # HuggingFace model
        DIM = 384
        METRIC = "IP"  # use cosine via normalized vectors

        # 1. Connect
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

        # 2. Ensure collection
        def ensure_collection(collection_name: str, dim: int, metric: str) -> Collection:
            if utility.has_collection(collection_name):
                return Collection(collection_name)

            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields, description="Text embeddings")
            col = Collection(name=collection_name, schema=schema, consistency_level="Strong")

            index_params = {
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
                "metric_type": metric,
            }
            col.create_index(field_name="embedding", index_params=index_params)
            return col

        col = ensure_collection(COLLECTION, DIM, METRIC)

        # 3. Get embeddings
        model = embedding_model.get_huggingface_embeddings(model_name=MODEL_NAME)
        texts = chunking.recursive_chunk_text()  # should return a list of strings
        vecs = model.encode(texts, convert_to_numpy=True).astype("float32")

        # 4. Normalize for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1, norms)

        # 5. Insert into Milvus (order must match schema: [text, embedding])
        col.insert([texts, vecs])
        col.flush()

        print("Inserted. Total entities:", col.num_entities)
        return col
