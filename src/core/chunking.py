import nltk
import langchain 
nltk.download('punkt_tab')
class file_chunking:     
       
    #Fixed-size chunking
    @staticmethod
    def overlap(text,chunk_size=500,overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    
    #Recursive chunking   
    @staticmethod
    def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 50):
        """
        Recursive chunking using LangChain.
        """
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError("Please install langchain: pip install langchain")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        return splitter.split_text(text)
    
    #Sentence-based chunking    
    @staticmethod
    def sentence_chunk(text):
        """
        Chunk text into sentences using NLTK's sent_tokenize.    
        Args:
        text (str): Input text string    
        Returns:
        list: List of sentences (chunks)
        """
        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            raise ImportError("Please install nltk: pip install nltk")        
        sentences = sent_tokenize(text)        
        return sentences
if __name__ == "__main__":
    sample_text = """Artificial Intelligence is transforming the world. 
    It is used in healthcare, finance, and education. 
    Sentence-based chunking helps in semantic search.
    The creation of a government grantmaking agency.
    They inspire curiosity about our own experiences and those of our neighbours.
    """
    print("Fixed Chunks:\n",file_chunking.overlap(sample_text,15,5))
    print("Recursive Chunks:\n",file_chunking.recursive_chunk(sample_text,15,5))
    print("Sentence Chunks:\n",file_chunking.sentence_chunk(sample_text))    
    
    ####output
    
    """
    Fixed Chunks:
 ['Artificial Inte', ' Intelligence i', 'nce is transfor', 'nsforming the w', 'the world. \n   ', ' \n    It is use', 's used in healt', 'healthcare, fin', ', finance, and ', ' and education.', 'tion. \n    Sent', ' Sentence-based', 'based chunking ', 'king helps in s', ' in semantic se', 'ic search.\n    ', '\n    The creati', 'reation of a go', ' a government g', 'ent grantmaking', 'aking agency.\n ', 'cy.\n    They in', 'ey inspire curi', ' curiosity abou', ' about our own ', ' own experience', 'iences and thos', ' those of our n', 'our neighbours.', 'ours.\n    ']
Recursive Chunks:
 ['Artificial', 'Intelligence', 'is', 'transforming', 'the world.', 'It is used', 'used in', 'in healthcare,', 'finance, and', 'and education.', 'Sentence-based', 'chunking helps', 'in semantic', 'search.', 'The', 'The creation', 'of a', 'a government', 'grantmaking', 'agency.', 'They', 'They inspire', 'curiosity', 'about our own', 'experiences', 'and those of', 'of our', 'neighbours.']
Sentence Chunks:
 ['Artificial Intelligence is transforming the world.', 'It is used in healthcare, finance, and education.', 'Sentence-based chunking helps in semantic search.', 'The creation of a government grantmaking agency.', 'They inspire curiosity about our own experiences and those of our neighbours.']
    """