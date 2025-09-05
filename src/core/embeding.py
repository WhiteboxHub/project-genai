
class embed_model:
    from typing import List, Union
    import numpy as np

    #SentenceTransformers
    @staticmethod
    def sentence_transformer_embed(texts: Union[str, List[str]], model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Generate embeddings using SentenceTransformers.
        Requires: pip install sentence-transformers
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence_transformers: pip install sentence_transformers")
        
    
        if isinstance(texts, str):
            texts = [texts]
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    #HuggingFace Transformer Embeddings
    @staticmethod
    def huggingface_embed(texts: Union[str, List[str]], model_name="distilbert-base-uncased"):
        """
        Generate embeddings using HuggingFace Transformers (CLS token pooling).
        Requires: pip install transformers torch
        """
        from transformers import AutoTokenizer, AutoModel
        import torch

        if isinstance(texts, str):
            texts = [texts]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    
    