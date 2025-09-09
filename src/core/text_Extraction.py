import PyPDF2

class extraction: 
    
    def readfile_pdf(self, pdf_path):
            """
            Extract raw text from a PDF file.
            """
            text = ""
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
    
    def readfile_txt(self, pdf_path):
         pass
    def readfile_csv(self, pdf_path):
         pass
    # def readfile(fileloca):  # .pdf , .txt ,.csv