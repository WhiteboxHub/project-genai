class file_chunking: 
    import nltk
    import langchain
    import PyPDF2
    import requests
    import bs4
    import csv

    #Fixed-size chunking
    @staticmethod
    def overlap(text,chunk_size=500,overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    
    #Recursive chunking (via LangChain)
    #Start with large chunks (e.g., sections), then recursively split into smaller chunks if too large.
    @staticmethod
    def recursive(text, chunk_size=500, overlap=50):
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError("Please install langchain: pip install langchain")
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)
    
    #Sentence-based chunking
    #Split text by sentences, then group multiple sentences until you reach a size limit.
    @staticmethod
    def sentence(text, max_tokens=200):
        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            raise ImportError("Please install nltk: pip install nltk")        
        sentences = sent_tokenize(text)
        chunks, current, tokens = [], [], 0
        for sent in sentences:
            sent_len = len(sent.split())
            if tokens + sent_len > max_tokens:
                chunks.append(" ".join(current))
                current, tokens = [], 0
            current.append(sent)
            tokens += sent_len
        if current:
            chunks.append(" ".join(current))
        return chunks
   
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
        