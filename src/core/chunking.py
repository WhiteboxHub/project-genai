from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


class Chunking:
    @staticmethod
    def fixed_length_chunking(ip_text,chunk_size):
        """
        fixed length chunking.
        arguments 
        1- input text
        2- chun size
        
        it returns a list of chunks
        """
    
        return [ip_text[i:i+chunk_size] for i in range(0,len(ip_text),chunk_size)]
    
    @staticmethod
    def overlap_chunking(ip_text,chunk_size,step_size):
        """
        Arguments
        1 input_text- input text that needs to be chunked
        2 chunk_size- The total number of tokens or characters that each chunk can
        contain.
        3 step_size- The number of tokens or characters that the window moves to
        create the next chunk, which defines the overlap.
        """
        chunks =[]
        for i in range(0,len(ip_text)-chunk_size+1,step_size):
            chunk = ip_text[i:i+chunk_size]
            chunks.append(chunk)
        return chunks
        

    # @staticmethod
    def recursive_chunking(ip_text,chunk_size,overlap):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap)
        return splitter.split_text(ip_text)


if __name__ == "__main__":
    sample_text = "This is the first sentence. Here is another one. And yet another one."
    print('fixed length --->',Chunking.fixed_length_chunking(sample_text,10))
    print('overlap --->',Chunking.overlap_chunking(sample_text,10,5))
    print('recursive --->',Chunking.recursive_chunking(sample_text,10,5))