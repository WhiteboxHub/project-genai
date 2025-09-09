#Load the data from the source
import PyPDF2
import csv
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader

class FileReader:
    """Utility class for reading different file types."""
    def read_text_from_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    '''
    def read_text_from_pdf(filepath):
        """Reads text from a PDF file."""
        text = ""
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text.replace('\n', ' ')
    '''
    @staticmethod
    def read_text_from_pdf(filepath):
        """Reads text from a PDF file using LangChain's PyPDFLoader."""
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        print(f"Number of document objects created: {len(docs)}")  # Add this line
        print(type(docs))  # Add this line
        print(docs[0])  # Add this line
        # print the last doc
        print(docs[-1])  # Add this line
        # Combine all page contents into a single string
        text = " ".join([doc.page_content for doc in docs])
        return text 

    def read_text_from_csv(filepath):
        """Reads text from a CSV file."""
        text = ""
        with open(filepath, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                text += ' '.join(row) + ' '
        return text.strip()