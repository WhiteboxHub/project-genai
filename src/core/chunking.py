
class file_chunking:
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
    def recrsive():
        # langchain
        pass
    
    
#  Sentence- Based chunking: splitting text based on sentence( use regex for splitting, does not depend on word token)
#  it does not break sentence for better semantic understanding,used in text summarization,QA, LLMs(eg chatgpt) 


    @staticmethod
    def sentence(text, chunk_size=500):
        """
        Split text into sentence sementic chunks using NLTK, where each chunk does not exceed max chunk_size 500.
        """
        import nltk
        from nltk.tokenize import sent_tokenize

        # Download the sentence tokenizer model (only needs to happen once)
        nltk.download('punkt', quiet=True)

        # Step 1: Tokenize text into sentences
        sentences = sent_tokenize(text)

        # Step 2: Group sentences into chunks
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding the next sentence doesn't exceed max_chunk_len
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        # Add any remaining sentence as a chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
        pass
    
    