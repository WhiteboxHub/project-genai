from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
from nltk.tokenize import sent_tokenize
from src.utils.logger import logger
# Make sure punkt tokenizer is available
nltk.download("punkt", quiet=True)
class text_chunking:

    @logger
    @staticmethod
    def overlap_chunking(text: str, chunk_size: int, overlap: int) -> list[str]:
        
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap  # move forward with overlap

        return chunks

    @logger
    @staticmethod
    def recursive_text_splitter(text: str, chunk_size: int, overlap: int) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # tries these in order
        )
        
        chunks = splitter.split_text(text)
        return chunks

    @logger
    @staticmethod
    def sentence(text : str):
        # nltk lib -- sentence chunking
        return sent_tokenize(text)
    
    