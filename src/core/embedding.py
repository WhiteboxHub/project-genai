from langchain_huggingface import HuggingFaceEmbeddings

class embedding_model:
    
    @staticmethod
    def get_huggingface_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes and returns a HuggingFaceEmbeddings object.

        Args:
            model_name (str): The name of the Hugging Face model to use for embeddings.

        Returns:
            HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings.
        """
        return HuggingFaceEmbeddings(model_name=model_name)