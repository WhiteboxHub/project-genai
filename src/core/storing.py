
from langchain_community.vectorstores import Chroma
from embeding import EmbedModel
from chunking import FileChunking


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
        
#==========OUTPUT===============================
#  🔹 Retrieved: This is a test document.
#  🔹 Retrieved: It has multiple sentences.
#  🔹 Retrieved: Useful for embeddings.       