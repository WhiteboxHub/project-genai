from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

class file_chunking:
    #Chunking without using any library- Owncode
    @staticmethod
    def overlap(text = str, chunk_size = int, overlap=int):
        """
        Splits text into overlapping chunks.

        :param text: Input string
        :param chunk_size: Length of each chunk
        :param overlap: Number of characters to overlap between chunks
        :return: List of text chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start = end - overlap  # move start back by overlap

        return chunks


   
    # Recursive Chunking using langchain  library  
    @staticmethod
    def recursive(text: str, chunk_size: int, overlap: int):
        """
        Recursive chunking using LangChain's RecursiveCharacterTextSplitter.
        
        :param text: Input string
        :param chunk_size: Max characters per chunk
        :param overlap: Overlap between chunks
        :return: List of chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            #separators=["\n\n", "\n", ". ", " ", ""],  # fallback hierarchy
            
        )
        chunks = splitter.split_text(text)
        return chunks

# Sentence chunking using NLTk library
    @staticmethod
    def sentence(text: str):

            """
            Splits text into chunks by sentences using NLTK.

            Args:
                text (str): The input text.

            Returns:
                List[str]: List of sentence chunks.
            """
            sentences = sent_tokenize(text)
            return sentences


    # ---------------- Example ---------------- #
if __name__ == "__main__":
    sample_text = """Retrieval-Augmented Generation (RAG) is a method 
    that combines retrieval and generation.
    It helps large language models access external knowledge. Sentence 
    chunking keeps full sentences intact."""
    chunks = file_chunking.overlap(sample_text, chunk_size=50, overlap=5)
    chunks1 = file_chunking.sentence(sample_text)
    chunks2 = file_chunking.recursive(sample_text, chunk_size=50, overlap=5)

    print("this is an example for chunking")
    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}: {c}")
    
    print("\n\nthis is an example for sentence chunking")
    for i, c in enumerate(chunks1, 1):
        print(f"Chunk {i}: {c}")

    print("\n\nThis is an example for recursive chunking")
    for i, c in enumerate(chunks2, 1):
        print(f"Chunk {i}: {c}")    

#print(file_chunking.overlap(sample_text,200,10))

     

     
    
    