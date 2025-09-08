
from PyPDF2 import PdfReader
import pandas as pd

def readfile(fileloc: str):
    """
    Reads text content from PDF, TXT, or CSV files.
    """
    print("Starting Readfile")
    if fileloc.endswith(".pdf"):
        text = ""
        reader = PdfReader(fileloc)
        print("Reader metadata:", reader.metadata)  # updated line
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif fileloc.endswith(".txt"):
        with open(fileloc, "r", encoding="utf-8") as f:
            return f.read()

    elif fileloc.endswith(".csv"):
        df = pd.read_csv(fileloc)
        return df.to_string()

    else:
        raise ValueError(f"Unsupported file type: {fileloc}")

