import csv
import requests
import bs4
class text_extraction:

# extract text from Pdf
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("Please install PyPDF2: pip install PyPDF2")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    
    # extract text from text file
    @staticmethod
    def extract_text_from_txt(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
        
    # extract text from csv file
    @staticmethod    
    def extract_text_from_csv(file_path):
        text = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                # Join all columns in a row into a single string
                text.append(" ".join(row))
            return "\n".join(text)

    # extract text from web
    @staticmethod
    def extract_text_from_web(url):
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch URL: {url}")
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Please install bs4: pip install bs4")        
        soup = BeautifulSoup(response.text, "html.parser")
        # Remove scripts/styles
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n")
        return text
if __name__ == "__main__":  
    print(extract_text_from_pdf("./Data/sciencerag.pdf"))

#output:
'''Photosynthesis in Plants
Photosynthesis is a biological process that occurs in plants, algae, and some bacteria, where light
energy is 
converted into chemical energy in the form of glucose (a sugar) and oxygen. This process is vital for
the survival 
of life on Earth, as it provides the base for the food chain and releases oxygen into the atmosphere.
### The Process of Photosynthesis..................
'''