from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
import PyPDF2

class file_chunking:
    @staticmethod
    def overlap(text: str, chunk_size: int, overlap:int)-> list[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if chunk_size <= overlap:
                raise ValueError("chunk_size must be greater than overlap to avoid an infinite loop.")
            
        chunks = []
        step = chunk_size - overlap
        i = 0
        
        while i < len(text):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
            i += step
        print(f"Number of chunks created: {len(chunks)}")
        print(f"Chunk type: {type(chunks)}")
        print("printing chunk[0]") 
        print(chunks[0])
        print("printing chunk[1]") 
        print(chunks[1])
        return chunks

    @staticmethod
    def recursive(text:str)->list[str]:
        # langchain
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 100,
            chunk_overlap = 0
         )
        result = splitter.split_text(text)
        print(f"Recursive chunking created {len(result)} chunks")

        print(f"Prinitng first 3 chunks:", result[0:3])

    @staticmethod
    def sentence_tokenize(text,sentences_per_chunk:int) -> list:
        # using nltk lib -- sentence chunking
        print("hello")
        sentences = sent_tokenize(text)
        print(len(sentences))
        print("sentences[0]")
        print(sentences[0])
        print("sentences[1]")
        print(sentences[1])
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= sentences_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining sentences
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"Created {len(chunks)} sentence-based chunks")
        return chunks
        

if __name__ == "__main__":
    def read_text_from_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
        
    def read_text_from_pdf(filepath):
        """Reads text from a PDF file."""
        text = ""
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text.replace('\n', ' ')
    '''
    def read_text_from_csv(filepath):
        """Reads text from a CSV file."""
        text = ""
        with open(filepath, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                text += ' '.join(row) + ' '
        return text.strip()
    
    '''
    # Example usage with your file_chunking class:
    text_from_file = read_text_from_file("Data/Gutenburg.txt")
    file_chunking.recursive(text_from_file)

    #file_chunking.overlap(text_from_file,300,50)
    #text = "hello,my name is Harman "
    #file_chunking.recursive(text)

    #text = "This is the first sentence. This is the second sentence. And here is the third one."
    #file_chunking.sentence_tokenize(text_from_file)
