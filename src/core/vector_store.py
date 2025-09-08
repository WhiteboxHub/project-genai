from langchain.vectorstores import Chroma

class vector_store:

    @staticmethod
    def create_chroma_vector_store(embeddings, persist_directory="./chroma_db"):
        """
        Initializes and returns a Chroma vector store.

        Args:
            embeddings: The embedding model to use for the vector store.
            persist_directory (str): The directory where the Chroma database will be persisted.

        Returns:
            Chroma: An instance of the Chroma vector store.
        """

        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    @staticmethod
    def pgvector_store():
        