#Load the data from the source

class file_load():
    @staticmethod
    def read_text_from_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
        
    def read_text_from_pdf(filepath):
        """Reads text from a PDF file."""
        text = ""
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text.replace('\n', ' ')
    '''
    def read_text_from_csv(filepath):
        """Reads text from a CSV file."""
        text = ""
        with open(filepath, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                text += ' '.join(row) + ' '
        return text.strip()
    '''