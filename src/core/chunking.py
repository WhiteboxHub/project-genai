
class file_chunking:
    @staticmethod
    def overlap(text, chunk_size, overlap):   
         
        
         assert 0 <= overlap < chunk_size, "overlap must be >=0 and < chunk_size"
         words = text.split()
         chunks = []
         start = 0
         n = len(words)
         step = chunk_size - overlap
         while start < n:
              end = min(start + chunk_size, n)
              chunks.append(" ".join(words[start:end]))
              start += step
         return chunks
     
     
text = "This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough."
word_chunks = overlap(text, chunk_size=5, overlap=2)


from langchain.text_splitter import RecursiveCharacterTextSplitter    
@staticmethod
def recursive(text):          
        
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=1,
)
        
        output = splitter.create_documents([text])
        print(output[:])
        
recursive_chunks = recursive(text = "Hi."

"I'm Harrison."

"How? Are? You?"
"Okay then f f f f."
"This is a weird text to write, but gotta test the splittingggg some how."

"Bye!"

"-H.")
print(recursive_chunks)
    
    
    
import nltk 
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
@staticmethod
def sentence(text, max_chars):
        # nltk lib -- sentence chunking 
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sent in sentences:
        if current_chunk and len(current_chunk) + len(sent) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = sent
        else:
            current_chunk = sent if not current_chunk else f"{current_chunk} {sent}"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

text = "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth. And finally, this is the fifth sentence."

    
sentence_chunks = sentence(text, max_chars=80)
sentence_chunks
    
    