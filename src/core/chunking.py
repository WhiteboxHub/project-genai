
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

#  Recursive character text splitter--------

    @staticmethod
    def overlap(text, chunk_size=100, overlap=20):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))
            i += chunk_size - overlap
        return chunks

    @staticmethod
    def sentence(text, max_chunk_len=500):
        import re
        raw_sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""

        for sentence in raw_sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_len:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    @staticmethod
    def recrsive(text, chunk_size=500, chunk_overlap=50):
        """
        Custom recursive-style chunker without LangChain.
        Tries to split by paragraphs, then sentences, then words.
        """
        import re

        def recursive_split(text, separators, chunk_size, chunk_overlap):
            if not separators:
                return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

            sep = separators[0]
            parts = text.split(sep)

            chunks = []
            current_chunk = ""

            for part in parts:
                if len(current_chunk) + len(part) + len(sep) <= chunk_size:
                    current_chunk += (sep + part) if current_chunk else part
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    if len(part) <= chunk_size:
                        current_chunk = part
                    else:
                        # Recurse with next separator
                        sub_chunks = recursive_split(part, separators[1:], chunk_size, chunk_overlap)
                        chunks.extend(sub_chunks)
                        current_chunk = ""

            if current_chunk:
                chunks.append(current_chunk.strip())

            # Add overlap
            if chunk_overlap > 0 and len(chunks) > 1:
                overlapped_chunks = []
                for i in range(len(chunks)):
                    start = max(0, i - 1)
                    overlap_chunk = " ".join(chunks[start:i + 1])
                    overlapped_chunks.append(overlap_chunk)
                return overlapped_chunks

            return chunks

        separators = ["\n\n", "\n", ".", " ", ""]
        return recursive_split(text, separators, chunk_size, chunk_overlap)
        
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
    
    