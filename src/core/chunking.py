
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize


class FileChunking:
    @staticmethod
    def overlap(text: str, chunk_size: int = 1000, overlap: int = 200):
        """
        Simple overlapping chunks.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def recursive(text: str, chunk_size: int = 1000, overlap: int = 200):
        """
        Recursive chunking using LangChain.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        return splitter.split_text(text)

    @staticmethod
    def sentence(text: str):
        """
        Sentence-based chunking using NLTK.
        """
        return sent_tokenize(text)


if __name__ == "__main__":
    sample_text = "This is the first sentence. Here is another one. And yet another one."
    print("🔹 Overlap:", FileChunking.overlap(sample_text, 20, 5))
    print("🔹 Recursive:", FileChunking.recursive(sample_text, 20, 5))
    print("🔹 Sentence:", FileChunking.sentence(sample_text))

#=======OUTPUT==========#
    
#[nltk_data] Downloading package punkt to /Users/Sona/nltk_data...
# [nltk_data]   Package punkt is already up-to-date!
# 🔹 Overlap: ['This is the first se', 'st sentence. Here is', 're is another one. A', 'ne. And yet another ', 'ther one.']
# 🔹 Recursive: ['This is the first', 'sentence. Here is', 'is another one. And', 'And yet another', 'one.']
# 🔹 Sentence: ['This is the first sentence.', 'Here is another one.', 'And yet another one.']