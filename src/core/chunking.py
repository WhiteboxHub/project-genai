from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
import nltk


class chunking:
    
    @staticmethod
    def chunk_text(text, chunk_size=1000, overlap=200):
        """
        Splits the input text into chunks of specified size with a given overlap.

        Args:
            text (str): The input text to be chunked.
            chunk_size (int): The size of each chunk.
            overlap (int): The number of overlapping characters between chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks
    
    @staticmethod
    def recursive_chunk_text(text, chunk_size=1000, overlap=200, min_chunk_size=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            min_chunk_size=min_chunk_size
        )
        return text_splitter.split_text(text)
    
    @staticmethod
    def character_chunk_text(text, chunk_size=1000, overlap=200):
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len
        )
        return text_splitter.split_text(text)
    
    @staticmethod
    def nltk_sentence_chunk_text(text, chunk_size=1000, overlap=200):

        nltk.download('punkt')
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                
                # Handle overlap
                while len(current_chunk) > chunk_size and overlap > 0:
                    overlap_part = current_chunk[:overlap]
                    chunks.append(overlap_part)
                    current_chunk = current_chunk[overlap:]
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks