
# from langchain_community.vectorstores import Chroma


# class ChromaDB:
#     def __init__(self, persist_dir: str = "./vectorstore"):
#         self.persist_dir = persist_dir

#     def store_data(self, texts):
#         """
#         Store embeddings into ChromaDB.
#         """
#         embeddings = EmbedModel.huggingface_embedding()
#         self.db = Chroma.from_texts(texts, embeddings, persist_directory=self.persist_dir)
#         self.db.persist()
#         return self.db

#     def retrieve_data(self, query: str, k: int = 3):
#         """
#         Retrieve top-k relevant chunks.
#         """
#         results = self.db.similarity_search(query, k=k)
#         return results


# if __name__ == "__main__":
#     text = "This is a test document. It has multiple sentences. Useful for embeddings."
#     chunks = FileChunking.sentence(text)

#     db = ChromaDB()
#     db.store_data(chunks)
#     res = db.retrieve_data("What is in the document?")
#     for r in res:
#         print("🔹 Retrieved:", r.page_content)
        
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
        base_collection_name: str = "docs_collection",
        embedding_model=None,
        dim: Optional[int] = None,
    ):
        """
        Connect to Milvus and create two collections:
        - COSINE search collection
        - IP (Inner Product) search collection
        """
        self.base_name = base_collection_name
        self.emb_model = embedding_model or EmbedModel.huggingface_embedding()

        # Wait for Milvus
        max_retries = 10
        for attempt in range(max_retries):
            try:
                connections.connect(alias="default", host=host, port=port)
                print(f"✅ Connected to Milvus at {host}:{port}")
                break
            except Exception:
                print(f"Milvus not ready, retrying ({attempt+1}/{max_retries})...")
                time.sleep(3)
        else:
            raise RuntimeError("Could not connect to Milvus after multiple attempts")

        # detect embedding dimension
        if dim is None:
            sample_vec = self.emb_model.embed_query("detect-dim")
            dim = len(sample_vec)
        self.dim = dim

        # collection names
        self.cosine_collection_name = f"{self.base_name}_cosine"
        self.ip_collection_name = f"{self.base_name}_ip"

        # initialize collections
        self.cosine_collection = self._init_collection(self.cosine_collection_name, metric_type="COSINE")
        self.ip_collection = self._init_collection(self.ip_collection_name, metric_type="IP")

    def _init_collection(self, name: str, metric_type: str):
        if not utility.has_collection(name):
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
            schema = CollectionSchema(fields, description=f"{metric_type} search collection")
            collection = Collection(name=name, schema=schema)

            # create index
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": metric_type,
                "params": {"nlist": 128},
            }
            collection.create_index(field_name="embedding", index_params=index_params)
        else:
            collection = Collection(name)
        collection.load()
        return collection

    def store_data(self, texts: List[str], filenames: Optional[List[str]] = None):
        if filenames is None:
            filenames = ["" for _ in texts]
        if len(filenames) != len(texts):
            raise ValueError("filenames and texts must have the same length")

        vectors = [self.emb_model.embed_query(t) for t in texts]
        entities = [filenames, texts, vectors]

        # insert into both collections
        self.cosine_collection.insert(entities)
        self.cosine_collection.flush()

        self.ip_collection.insert(entities)
        self.ip_collection.flush()

        return {"inserted": len(texts)}

    # ----------------- Separate search functions -----------------
    def ann_cosine_search(self, query: str, k: int = 3, nprobe: int = 10):
        query_vector = self.emb_model.embed_query(query)
        return self._search_collection(self.cosine_collection, query_vector, k, nprobe, metric_type="COSINE")

    def ip_search(self, query: str, k: int = 3, nprobe: int = 10):
        query_vector = self.emb_model.embed_query(query)
        return self._search_collection(self.ip_collection, query_vector, k, nprobe, metric_type="IP")

    def _search_collection(self, collection, query_vector, k, nprobe, metric_type):
        search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["filename", "text"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "filename": hit.entity.get("filename"),
                "text": hit.entity.get("text"),
                "score": hit.score,
            })
        return hits


# ---------------- Example ----------------
if __name__ == "__main__":
    from chunking import FileChunking

    text = "This is a test document. It has multiple sentences."
    chunks = FileChunking.sentence(text)
    filenames = ["example.pdf"] * len(chunks)

    db = MilvusDB()
    db.store_data(chunks, filenames)

    print("\n--- COSINE Search ---")
    res_cos = db.ann_cosine_search("test document")
    for r in res_cos:
        print(r)

    print("\n--- IP Search ---")
    res_ip = db.ip_search("test document")
    for r in res_ip:
        print(r)
#✅ Connected to Milvus at localhost:19530

# --- COSINE Search ---
# {'filename': 'example.pdf', 'text': 'This is a test document.', 'score': 0.7600330710411072}
# {'filename': 'example.pdf', 'text': 'This is a test document.', 'score': 0.7600328922271729}
# {'filename': 'langchain.pdf', 'text': 'Abstract—Due to the unstructured nature of the PDF \ndocument format and the requirement for precise and pertinent \nsearch results, querying a PDF can take time and effort.', 'score': 0.3748253583908081}

# --- IP Search ---
# {'filename': 'example.pdf', 'text': 'This is a test document.', 'score': 0.7600330114364624}
# {'filename': 'example.pdf', 'text': 'This is a test document.', 'score': 0.7600328326225281}
# {'filename': 'langchain.pdf', 'text': 'Abstract—Due to the unstructured nature of the PDF \ndocument format and the requirement for precise and pertinent \nsearch results, querying a PDF can take time and effort.', 'score': 0.3748253583908081}