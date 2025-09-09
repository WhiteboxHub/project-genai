
from sentence_transformers import SentenceTransformer
from chunking_pdf import PDFChunker


class embed_model_pdf:

    @staticmethod
    def sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
            """
            Load a SentenceTransformer embedding model.
            
            :param model_name: Pretrained model from sentence-transformers
            :return: Embedding function
            """
            model = SentenceTransformer(model_name)

            def embed(texts):   #its a closure function- return embed point to the given text
                if isinstance(texts, str):
                    texts = [texts]
                return model.encode(texts, convert_to_numpy=True)
            
            return embed
    

    # ---------------- Example Usage ----------------
    
if __name__ == "__main__":

    #pdf_file = "/Users/hema/Desktop/project-genai/Data/agentic-ai.pdf"  
   # pdf_file = "/Users/hema/Desktop/project-genai/Data/chunking_strategies.pdf"
    pdf_file = "/Users/hema/Desktop/project-genai/Data/langchain.pdf"

    chunker = PDFChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_pdf(pdf_file)

     #to show the chunks
    print(f"✅ Extracted {len(chunks)} chunks from {pdf_file}\n")
    for i, c in enumerate(chunks[:5], 1):  # show first 5 chunks
        print(f"Chunk {i}:\n{c}\n{'-'*50}") # prints chunk and separator as 50 '-'


    # SentenceTransformer
    st_embed = embed_model_pdf.sentence_transformer()
    print("SentenceTransformer:", st_embed(chunks))

    #-------------------------------------------OUTPUT----------------------------
"""
    SentenceTransformer: [[-4.14562151e-02  3.53128389e-02 -3.91984619e-02 ... -6.18018508e-02
  -1.56648643e-02  1.01850241e-01]
 [-4.65343781e-02 -7.40164774e-04 -6.07665665e-02 ... -4.05293033e-02
  -7.13084498e-03  1.54373348e-02]
 [-5.25530614e-02  4.20908406e-02 -7.83482119e-02 ... -3.40263068e-05
  -7.17761368e-02  5.02889901e-02]
 ...
 [ 2.08277348e-02 -8.28967371e-04 -1.11460820e-01 ... -7.76207596e-02
   9.61020514e-02  2.79624425e-02]
 [-2.79041938e-02  3.44636887e-02  3.38561125e-02 ... -8.05080533e-02
  -4.86741625e-02  3.57083753e-02]
 [ 1.11205585e-01  5.47756851e-02  1.87006630e-02 ... -9.26225632e-02
  -8.22639614e-02 -2.40598712e-02]]"""