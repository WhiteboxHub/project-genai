
from langchain_community.vectorstores import Chroma


class ChromaDB:
    def __init__(self, persist_dir: str = "./vectorstore"):
        self.persist_dir = persist_dir

    def store_data(self, texts):
        """
        Store embeddings into ChromaDB.
        """
        embeddings = EmbedModel.huggingface_embedding()
        self.db = Chroma.from_texts(texts, embeddings, persist_directory=self.persist_dir)
        self.db.persist()
        return self.db

    def retrieve_data(self, query: str, k: int = 3):
        """
        Retrieve top-k relevant chunks.
        """
        results = self.db.similarity_search(query, k=k)
        return results


if __name__ == "__main__":
    text = "This is a test document. It has multiple sentences. Useful for embeddings."
    chunks = FileChunking.sentence(text)

    db = ChromaDB()
    db.store_data(chunks)
    res = db.retrieve_data("What is in the document?")
    for r in res:
        print("🔹 Retrieved:", r.page_content)
        
# #==========OUTPUT===============================
# #  🔹 Retrieved: This is a test document.
# #  🔹 Retrieved: It has multiple sentences.
# #  🔹 Retrieved: Useful for embeddings.       
# #===============================================

import time
from typing import List, Optional
from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)
from embeding import EmbedModel


class MilvusDB:
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "docs_collection",
        embedding_model=None,
        dim: Optional[int] = None,
    ):
        """
        Connect to Milvus and ensure a collection exists.
        Waits until Milvus server is available.
        """
        self.collection_name = collection_name
        # initialize embedding model
        self.emb_model = embedding_model or EmbedModel.huggingface_embedding()

        # --------- Wait for Milvus to be ready ----------
        max_retries = 10
        for attempt in range(max_retries):
            try:
                connections.connect(alias="default", host=host, port=port)
                print(f"Connected to Milvus at {host}:{port}")
                break
            except Exception as e:
                print(f"Milvus not ready, retrying ({attempt+1}/{max_retries})...")
                time.sleep(3)
        else:
            raise RuntimeError("Could not connect to Milvus after multiple attempts")
        # ------------------------------------------------

        # detect embedding dimension if not provided
        if dim is None:
            sample_vec = self.emb_model.embed_query("detect-dim")
            dim = len(sample_vec)
        self.dim = dim

        # create collection if missing
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
            schema = CollectionSchema(fields, description="Document chunks with filename and embeddings")
            self.collection = Collection(name=self.collection_name, schema=schema)

            # create an index for ANN search
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128},
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self.collection = Collection(self.collection_name)

        # load the collection into memory
        self.collection.load()

    def store_data(self, texts: List[str], filenames: Optional[List[str]] = None):
        if filenames is None:
            filenames = ["" for _ in texts]
        if len(filenames) != len(texts):
            raise ValueError("filenames and texts must have the same length")

        # compute embeddings
        vectors = [self.emb_model.embed_query(t) for t in texts]

        # insert into Milvus
        entities = [filenames, texts, vectors]  # order matches schema fields (excluding auto pk)
        self.collection.insert(entities)
        self.collection.flush()
        return {"inserted": len(texts)}

    def retrieve_data(self, query: str, k: int = 3):
        query_vector = self.emb_model.embed_query(query)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["filename", "text"],
        )

        hits = []
        for hit in results[0]:
            fname = hit.entity.get("filename")
            text = hit.entity.get("text")
            score = hit.score
            hits.append({"filename": fname, "text": text, "score": score})
        return hits


# Example usage
if __name__ == "__main__":
    from chunking import FileChunking

    text = "This is a test document. It has multiple sentences. Useful for embeddings."
    chunks = FileChunking.sentence(text)
    filenames = ["example.pdf"] * len(chunks)

    db = MilvusDB()
    print(db.store_data(chunks, filenames))

    res = db.retrieve_data("What is in the document?", k=3)
    for r in res:
        print("Retrieved:", r)

