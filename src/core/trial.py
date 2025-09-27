from storing import MilvusDB
from embeding import EmbedModel

try:
    db = MilvusDB(embedding_model=EmbedModel.huggingface_embedding())
    print("Connected to Milvus successfully!")
except Exception as e:
    print("Milvus connection failed:", e)