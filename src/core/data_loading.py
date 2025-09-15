from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def load_documents(file_path):
    """
    Load and split text documents from a file.

    Args:
        file_path (str): Path to the text file.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: A list of chunked Document objects.
    """
    loader = TextLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)
