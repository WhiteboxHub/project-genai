from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbedModel:
    @staticmethod
    def sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
        """
        Use SentenceTransformers directly.
        """
        model = SentenceTransformer(model_name)
        return model

    @staticmethod
    def huggingface_embedding(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Use HuggingFace embedding via LangChain.
        """
        return HuggingFaceEmbeddings(model_name=model_name)


if __name__ == "__main__":
    model = EmbedModel.sentence_transformer()
    print("🔹 SentenceTransformer embedding:", model.encode("Hello world"))

    hf_embed = EmbedModel.huggingface_embedding()
    print("🔹 HuggingFace embedding:", hf_embed.embed_query("Hello world"))

#================output==================
#     🔹 SentenceTransformer embedding: [-3.44773196e-02  3.10232081e-02  6.73497515e-03  2.61089951e-02
#  -3.93620133e-02 -1.60302445e-01  6.69240057e-02 -6.44143531e-03
#  -4.74504307e-02  1.47588588e-02  7.08752647e-02  5.55276386e-02
#   1.91933624e-02 -2.62513310e-02 -1.01095568e-02 -2.69404389e-02
#   2.23074351e-02 -2.22266428e-02 -1.49692580e-01 -1.74929798e-02
#   7.67624564e-03  5.43522686e-02  3.25444061e-03  3.17258686e-02
#  -8.46214369e-02 -2.94060037e-02  5.15955612e-02  4.81240377e-02