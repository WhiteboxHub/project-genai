from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader

def load_pdf_with_langchain(filepath):
    loader = PyMuPDFLoader(filepath)
    documents = loader.load()
    return documents

'''res=load_pdf_with_langchain(r"Data\\Medical_book.pdf")
print(res[1])'''

def load_csv_with_langchain(filepath):
    loader=CSVLoader(filepath)
    document=loader.load()
    return document

'''res2=load_csv_with_langchain(r"Data\\creditcard.csv")
for doc in res2[:3]:
    print(doc.page_content)'''