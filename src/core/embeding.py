from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

class model_embed:


    @staticmethod
    def encode_sentence_transform():
        """
         Class to generate embedding using SentenceTransformer directly
        """
        sentences = ["Hi this is first sentence", "LangChain and Hugging Face embeddings!"]

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        embeddings = model.encode(sentences, convert_to_numpy=True).tolist()
        print("SentenceTransformer embeddings:")
        print(embeddings)
        return embeddings

    @staticmethod
    def langchain_huggingface_embeddings():
        """
        Generate embeddings using LangChain HuggingFaceEmbeddings wrapper.
        """
        sentences = ["Hi this is first sentence", "LangChain and Hugging Face embeddings!"]

        # LangChain wrapper for Hugging Face sentence transformers
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

        # Use embed_query for single sentence, or loop for multiple
        embeddings = [embedding_model.embed_query(sentence) for sentence in sentences]
        print("HuggingFace embeddings (via LangChain):")
        print(embeddings)
        return embeddings


# Example usage
if __name__ == "__main__":
    model_embed.encode_sentence_transform()
    model_embed.langchain_huggingface_embeddings()
