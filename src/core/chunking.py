
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
nltk.download("punkt")  # Correct spelling
from nltk.tokenize import sent_tokenize

class FileChunking:
    @staticmethod
    def overlap(text: str, chunk_size: int, overlap: int):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def recursive(text: str, chunk_size: int, overlap: int):
        # Recursive chunking using LangChain
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        return splitter.split_text(text)

    @staticmethod
    def sentence(text: str):
        # NLTK sentence chunking
        return sent_tokenize(text)

# Main block should be outside the class
if __name__ == "__main__":
    sample_text = "This is my first sentence. Here is my new one. It is better than the old one."

    print(" * Overlap:", FileChunking.overlap(sample_text, 20, 5))
    print(" * Recursive:", FileChunking.recursive(sample_text, 20, 5))
    print(" * Sentence:", FileChunking.sentence(sample_text))