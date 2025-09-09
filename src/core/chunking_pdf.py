
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

#from text_Extraction import extraction

class PDFChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]  # hierarchy of splits
        )
     
    def extract_text(self, pdf_path):
       
       # Extract raw text from a PDF file.
       
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
            

    def chunk_pdf(self, pdf_file):
        """
        Extract text and split into chunks.
            """
        #pdf_file = "/Users/hema/Desktop/project-genai/Data/agentic-ai.pdf"  
        # pdf_file = "/Users/hema/Desktop/project-genai/Data/chunking_strategies.pdf"

        pdf_file = "/Users/hema/Desktop/project-genai/Data/langchain.pdf"

        # text = extraction.readfile_pdf(pdf_file)
        text = PDFChunker.extract_text(pdf_file)
        chunks = self.splitter.split_text(text)

        return chunks

   
# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    #pdf_file = "/Users/hema/Desktop/project-genai/Data/agentic-ai.pdf"  
   # pdf_file = "/Users/hema/Desktop/project-genai/Data/chunking_strategies.pdf"
    pdf_file = "/Users/hema/Desktop/project-genai/Data/langchain.pdf"

    chunker = PDFChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_pdf(pdf_file)

    print(f"✅ Extracted {len(chunks)} chunks from {pdf_file}\n")
    for i, c in enumerate(chunks[:5], 1):  # show first 5 chunks
        print(f"Chunk {i}:\n{c}\n{'-'*50}") # prints chunk and separator as 50 '-'

#-------------------output-------------------------------------
"""
Sample output FOR "agentic-ai.pdf"
✅ Extracted 205 chunks from /Users/hema/Desktop/project-genai/Data/agentic-ai.pdf

Chunk 1:
Agentic AI 
– the new 
frontier in 
GenAI 
An executive playbook 
Harnessing AI isn’t just about 
technology— it’s about unleashing 
unprecedented potential. 
In an era where speed, eﬃciency, and customer centricity dictate market leadership, organisations need to
--------------------------------------------------
Chunk 2:
harness every tool at their disposal. Over the past couple of years, artiﬁcial intelligence (AI) has exploded onto 
the world stage, with companies and individuals across the globe rapidly adopting the technology. The GCC is
--------------------------------------------------

---------Sample output FOR "chunking_strategies.pdf"

Chunk 1:
See discussions, st ats, and author pr ofiles f or this public ation at : https://www .researchgate.ne t/public ation/308158087
Chu nking mechanisms and learning
Article  · Januar y 2012
CITATIONS
47READS
20,621
2 author s, including:
Fernand Gobe t
--------------------------------------------------
Chunk 2:
20,621
2 author s, including:
Fernand Gobe t
London School of Ec onomics and P olitic al Scienc e
422 PUBLICA TIONS    13,650  CITATIONS    
SEE PROFILE
All c ontent f ollo wing this p age was uplo aded b y Fernand Gobe t on 14 A ugust 2017.


------output for "langchain.pdf"
Chunk 1:
See discussions, st ats, and author pr ofiles f or this public ation at : https://www .researchgate.ne t/public ation/372529063
An Effective Query System Using LLMs and LangChain
Article    in  International Journal of Engineering R esearch and  · July 2023
CITATIONS
11READS
3,834
2 author s:
--------------------------------------------------
Chunk 2:
CITATIONS
11READS
3,834
2 author s:
Adith Sr eeram a
VIT-AP Univ ersity
2 PUBLICA TIONS    11 CITATIONS    
SEE PROFILE
Jithendr a Sai P appuri
Geor ge Mason Univ ersity
3 PUBLICA TIONS    11 CITATIONS    
SEE PROFILE
--------------------------------------------------
"""