
"""
Chunking with LangChain + Hugging Face + SentenceTransformers
Supports:
    - Overlap chunking
    - Recursive chunking
    - Embedding with SentenceTransformer
    - Summarization with Hugging Face T5Gemma
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# --------------------------
# Load Models
# --------------------------

# Hugging Face T5Gemma for summarization
GEN_MODEL_NAME = "google/t5gemma-b-b-prefixlm"
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
summarizer = pipeline("summarization", model=gen_model, tokenizer=gen_tokenizer)

# SentenceTransformer for embeddings
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMB_MODEL_NAME)

# --------------------------
# Chunking with LangChain
# --------------------------

def overlap_chunking(text, chunk_size=200, chunk_overlap=50):
    """
    Overlap chunking with LangChain's CharacterTextSplitter
    """
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def recursive_chunking(text, chunk_size=200, chunk_overlap=50):
    """
    Recursive chunking with LangChain's RecursiveCharacterTextSplitter
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # try larger splits first
    )
    return splitter.split_text(text)

# --------------------------
# Embedding + Chunking Workflow
# --------------------------

def embed_chunks(chunks):
    """
    Generate embeddings for each chunk
    """
    return embedder.encode(chunks, convert_to_tensor=True)


def summarize_chunk(chunk, max_length=100):
    """
    Summarize a single chunk with T5Gemma
    """
    summary = summarizer(chunk, max_length=max_length, min_length=20, do_sample=False)
    return summary[0]["summary_text"]


# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    text = (
        "Architects play a crucial role in shaping the built environment. "
        "They are responsible for designing spaces that are not only functional but also aesthetically pleasing. "
        "Throughout history, architects like Frank Lloyd Wright, Le Corbusier, and Zaha Hadid have introduced groundbreaking ideas that transformed how people experience buildings. "
        "Modern architecture emphasizes sustainability, with many architects now focusing on eco-friendly materials, energy efficiency, and designs that harmonize with nature. "
        "Beyond technical skills, architects must also collaborate with engineers, city planners, and clients to bring their visions to life. "
        "The profession blends art, science, and social responsibility, making architecture one of the most influential fields in shaping human life."
    )

    print("\n--- Overlap Chunking ---")
    overlap_chunks = overlap_chunking(text, chunk_size=80, chunk_overlap=20)
    for i, ch in enumerate(overlap_chunks, 1):
        print(f"Chunk {i}: {ch}")

    print("\n--- Recursive Chunking ---")
    recursive_chunks = recursive_chunking(text, chunk_size=80, chunk_overlap=20)
    for i, ch in enumerate(recursive_chunks, 1):
        print(f"Chunk {i}: {ch}")

    print("\n--- Embeddings ---")
    embeddings = embed_chunks(overlap_chunks)
    print("Embeddings shape:", embeddings.shape)

    print("\n--- Summarization ---")
    for i, ch in enumerate(overlap_chunks[:2], 1):  # summarize first 2 chunks
        summary = summarize_chunk(ch)
        print(f"Summary {i}: {summary}")


        # # handles chunking of long text into smaller pieces that can fit into the model’s max token length (512 tokens for T5).
# Why Use LangChain Here?
# Prebuilt Text Splitters
# LangChain provides CharacterTextSplitter (for overlap) and RecursiveCharacterTextSplitter (for recursive).
# These are battle-tested utilities — you don’t have to reimplement splitting logic.
# For example, recursive splitter automatically tries to split on \n\n, \n, . before chopping words, so chunks stay natural.
# Consistency with Vector DBs
# LangChain integrates directly with FAISS, Chroma, Pinecone, Weaviate, etc.
# So once you chunk text, you can push embeddings into a vector store with just a few lines.
# Pipeline Abstraction
# LangChain is designed for RAG workflows.
# LangChain isn’t strictly required  here.just makes chunking and scaling easier, especially if you’re planning to go toward a full RAG pipeline.
# --------------------------
# Langchain implimented
# --------------------------

















































































# --------------------------
# No langchain implimented here..only simple scripts
# --------------------------

# """Chunking utilities with Hugging Face T5Gemma model and SentenceTransformer embeddings.
# Supports:
#     - Overlap chunking
#     - Recursive chunking
# """

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from sentence_transformers import SentenceTransformer
# import textwrap

# # --------------------------
# # Load Models
# # --------------------------

# # Hugging Face T5Gemma for generation/summarization
# GEN_MODEL_NAME = "google/t5gemma-b-b-prefixlm"
# gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
# gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
# summarizer = pipeline("summarization", model=gen_model, tokenizer=gen_tokenizer)

# # SentenceTransformer for embeddings
# EMB_MODEL_NAME = "all-MiniLM-L6-v2"
# embedder = SentenceTransformer(EMB_MODEL_NAME)

# # --------------------------
# # Chunking Methods
# # --------------------------

# def overlap_chunking(text, chunk_size=500, overlap=100):
#     """
#     Split text into overlapping chunks.
#     """
#     words = text.split()
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = start + chunk_size
#         chunk = " ".join(words[start:end])
#         chunks.append(chunk)
#         start += (chunk_size - overlap)  # move forward with overlap
#     return chunks


# def recursive_chunking(text, max_chunk_size=500):
#     """
#     Recursively split text by paragraphs -> sentences -> smaller units
#     to ensure chunks do not exceed max_chunk_size.
#     """
#     if len(text.split()) <= max_chunk_size:
#         return [text]

#     # First try splitting by paragraphs
#     if "\n" in text:
#         parts = text.split("\n")
#     else:
#         # Fall back to splitting by sentences
#         parts = text.split(". ")

#     chunks = []
#     for part in parts:
#         if len(part.split()) > max_chunk_size:
#             # Recurse until small enough
#             chunks.extend(recursive_chunking(part, max_chunk_size))
#         else:
#             chunks.append(part.strip())
#     return [c for c in chunks if c]


# # --------------------------
# # Embedding + Chunking Workflow
# # --------------------------

# def embed_chunks(chunks):
#     """
#     Generate embeddings for each chunk.
#     """
#     return embedder.encode(chunks, convert_to_tensor=True)


# def summarize_chunk(chunk, max_length=100):
#     """
#     Summarize a single chunk with T5Gemma.
#     """
#     summary = summarizer(chunk, max_length=max_length, min_length=20, do_sample=False)
#     return summary[0]['summary_text']


# # --------------------------
# # Example Usage
# # --------------------------

# if __name__ == "__main__":
#     text = (
#         "Architects play a crucial role in shaping the built environment. "
#     "They are responsible for designing spaces that are not only functional but also aesthetically pleasing. "
#     "Throughout history, architects like Frank Lloyd Wright, Le Corbusier, and Zaha Hadid have introduced groundbreaking ideas that transformed how people experience buildings. "
#     "Modern architecture emphasizes sustainability, with many architects now focusing on eco-friendly materials, energy efficiency, and designs that harmonize with nature. "
#     "Beyond technical skills, architects must also collaborate with engineers, city planners, and clients to bring their visions to life. "
#     "The profession blends art, science, and social responsibility, making architecture one of the most influential fields in shaping human life."
# )

#     print("\n--- Overlap Chunking ---")
#     overlap_chunks = overlap_chunking(text, chunk_size=15, overlap=5)
#     for i, ch in enumerate(overlap_chunks, 1):
#         print(f"Chunk {i}: {ch}")

#     print("\n--- Recursive Chunking ---")
#     recursive_chunks = recursive_chunking(text, max_chunk_size=15)
#     for i, ch in enumerate(recursive_chunks, 1):
#         print(f"Chunk {i}: {ch}")

#     print("\n--- Embeddings ---")
#     embeddings = embed_chunks(overlap_chunks)
#     print("Embeddings shape:", embeddings.shape)

#     print("\n--- Summarization ---")
#     for i, ch in enumerate(overlap_chunks[:2], 1):  # summarize first 2 chunks
#         summary = summarize_chunk(ch)
#         print(f"Summary {i}: {summary}")






























