from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
from nltk.tokenize import sent_tokenize

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


# Example usage
    """ 
    if __name__ == "__main__":
        text = "This is a simple example to demonstrate chunking with overlap."
        chunks = file_chunking.overlap(text, chunk_size=15, overlap=5)
        for i, c in enumerate(chunks, 1):
            print(f"Chunk {i}: {c}")
    """

    # Chunking using langchain  library  
    @staticmethod
    def recursive(text: str, chunk_size: int = 200, overlap: int = 50):
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
            separators=["\n\n", "\n", ". ", " ", ""],  # fallback hierarchy
        )
        chunks = splitter.split_text(text)
        return chunks


    """# ---------------- Example Usage ----------------
    if __name__ == "__main__":
        text = (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances "
            "the capabilities of large language , models by allowing them to retrieve "
            "relevant documents from an external knowledge base during inference. "
            "This helps reduce hallucinations and improves factual accuracy."
        )

        chunks = file_chunking.recursive(text, chunk_size=50, overlap=10)

        for i, c in enumerate(chunks, 1):
            print(f"🔹 Chunk {i}: {c}")
    """       

    @staticmethod
    def sentence(text: str):
            nltk.download('punkt_tab')
            
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
    sample_text = """Retrieval-Augmented Generation (RAG) is a method that combines retrieval and generation.
    It helps large language models access external knowledge. Sentence chunking keeps full sentences intact."""

    chunks = file_chunking.sentence(sample_text)

    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}: {c}")

     
    
    