import os
import csv
import fitz   
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download('punkt', quiet=True)

class file_chunking:
    @staticmethod
    def read_file(file_path):
        """
        Reads text from .txt, .csv, or .pdf files
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[-1].lower()
        text = ""

        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                text = "\n".join([",".join(row) for row in reader])

        elif ext == ".pdf":
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text("text")

        else:
            raise ValueError("Unsupported file format. Use .txt, .csv, or .pdf")

        if not text.strip():
            raise ValueError("File is empty or contains no readable text.")

        return text.strip()

    @staticmethod
    def overlap(text, chunk_size=200, overlap=50):
        """
        Split text into overlapping chunks
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    @staticmethod
    def recursive(text, chunk_size=200, chunk_overlap=50):
        """
        Recursive chunking using LangChain's RecursiveCharacterTextSplitter
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

    @staticmethod
    def sentence(text):
        """
        Sentence-based chunking using NLTK
        """
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download('punkt', quiet=True)
        return nltk.sent_tokenize(text)


if __name__ == "__main__":
    
    file_path = r"C:\Users\aknar\OneDrive\Documents\Desktop\ai-agent\project-genai\Data\sample.txt"

    text = file_chunking.read_file(file_path)

    print("First 3 Overlap Chunks:")
    print(file_chunking.overlap(text, chunk_size=50, overlap=10)[:3])

    print("\nFirst 3 Recursive Chunks:")
    print(file_chunking.recursive(text, chunk_size=50, chunk_overlap=10)[:3])

    print("\nFirst 3 Sentences:")
    print(file_chunking.sentence(text)[:3])