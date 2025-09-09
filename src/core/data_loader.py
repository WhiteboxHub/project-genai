from langchain_community.document_loaders import PyMuPDFLoader , DirectoryLoader

def pdf_docs(file_path):
    """Load a PDF file and return its documents."""
    file_path= "Data/pdf"
    dir_loader= DirectoryLoader(file_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader , loader_kwargs={"sort": True} )

    documents = dir_loader.load()
    return documents