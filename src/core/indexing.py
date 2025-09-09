# from langchain_community.document_loaders import PyPDFLoader
# from pathlib import Path

# def load_pdf(file_path):
#     """Load a PDF file and return its documents."""
#     loader = PyPDFLoader(str(file_path))
#     documents = loader.load()
#     return documents

# if __name__ == "__main__":
#     # Define file paths using pathlib for better cross-platform compatibility
#     file_paths = [
#         Path("C:/Users/GursewakNeet/Documents/project-genai/Data/model_S_owners_manual.pdf"),
#         Path("C:/Users/GursewakNeet/Documents/project-genai/Data/model_X_owners_manual.pdf")
#     ]
#     for path in file_paths:
#         docs = load_pdf(path)
#         print(f"Loaded {len(docs)} documents from {path}")
