from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
from nltk.tokenize import sent_tokenize
import os
from dotenv import load_dotenv

load_dotenv()


class file_chunking:
    nltk.download('punkt_tab')
    @staticmethod
    def overlap(text, chunk_size= 100, overlap=20):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))
            i += chunk_size - overlap
        return chunks

        pass


    
    @staticmethod
    def recursive(text: str, chunk_size: int = 500, overlap: int = 50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        return splitter.split_text(text)

   
       
    @staticmethod
    def sentence(text,sentences_per_chunk=2):
        sentences=sent_tokenize(text)
        return [' '.join(sentences[i:i+sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]


 



'''file_path = os.getenv("file_path")
with open(file_path, "r") as file:
    text = file.read()
chunks = file_chunking.recrsive(text, chunk_size=100, overlap=20)
chunks2=file_chunking.overlap(text,chunk_size=50,overlap=5)
print(len(chunks))
print(len(chunks2))
sentence1=file_chunking.sentence(text)
print(sentence1[0])'''



      
        
    
    