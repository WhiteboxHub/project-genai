
import os
from chunking import FileChunking
from embeding import EmbedModel
from storing import MilvusDB
from text_extraction import readfile

# Get the directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Decision_Trees_Random_Forests_Notes.pdf")


# --- Debug: current directory ---
print("Current working directory:", os.getcwd())
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# --- Initialize Milvus ---
print("Initializing Milvus...")
db = MilvusDB(embedding_model=EmbedModel.huggingface_embedding())

# --- Read file ---
print(f"Reading file: {file_path}")
text = readfile(file_path).strip()
print(f"Extracted {len(text)} characters")

if not text:
    print(f"Skipped empty file: {file_path}")
else:
    # --- Chunk text ---
    chunks = FileChunking.sentence(text)
    print(f"Created {len(chunks)} chunks")

    # --- Prepare metadata ---
    filenames = [file_path] * len(chunks)

    # --- Store in Milvus ---
    print("Storing data in Milvus...")
    inserted_info = db.store_data(chunks, filenames)
    print(f"Stored {inserted_info['inserted']} chunks from {file_path}")

    # --- Optional: test retrieval ---
    query = "What is in the document?"
    print(f"Testing retrieval for query: {query}")
    results = db.retrieve_data(query, k=3)
    for r in results:
        print("Retrieved:", r)
#==============OUTPUT==================
#         Starting Readfile
# Reader metadata: {'/Author': '(anonymous)', '/CreationDate': "D:20250821233034+00'00'", '/Creator': '(unspecified)', '/Keywords': '', '/ModDate': "D:20250821233034+00'00'", '/Producer': 'ReportLab PDF Library - www.reportlab.com', '/Subject': '(unspecified)', '/Title': '(anonymous)', '/Trapped': '/False'}
# Extracted 3312 characters
# Created 58 chunks
# Storing data in Milvus...
# Stored 58 chunks from /Users/Sona/Documents/whitebox/project-genai/src/core/Decision_Trees_Random_Forests_Notes.pdf
# Testing retrieval for query: What is in the document?
# Retrieved: {'filename': '/Users/Sona/Documents/whitebox/project-genai/src/core/Decision_Trees_Random_Forests_Notes.pdf', 'text': 'Slide 2: Learning Objectives\nExplain what students will learn: 1.', 'score': 0.30141007900238037}
# Retrieved: {'filename': '/Users/Sona/Documents/whitebox/project-genai/src/core/Decision_Trees_Random_Forests_Notes.pdf', 'text': 'Slide 3: What is a Decision Tree?', 'score': 0.2719855010509491}
# Retrieved: {'filename': '/Users/Sona/Documents/whitebox/project-genai/src/core/Decision_Trees_Random_Forests_Notes.pdf', 'text': 'Emphasize that this session will cover both conceptual understanding and\npractical implementation in Python.', 'score': 0.26902082562446594}
