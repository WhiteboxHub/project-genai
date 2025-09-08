from datasets import load_dataset
import PyPDF2
from src.utils.logger import logger
import os

class load_file_loading:

    @logger
    @staticmethod
    def pdf_reader(file_path : str):
        try:
            # Open the PDF file in read-binary mode
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                # Loop through all pages
                for page in reader.pages:
                    text += page.extract_text()

            return text
        except FileNotFoundError as e:
            print("file not found")
            raise
    
    
    @logger
    @staticmethod
    def txt_reader(file_path : str):
        try:
            # Open the PDF file in read-binary mode
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                # Loop through all pages
                for page in reader.pages:
                    text += page.extract_text()

            return text
        except FileNotFoundError as e:
            print("file not found")
            raise
        except Exception as e:
            print(f"Unknown exception occured {e}")
            raise


class Data_loading:
    
    @logger
    @staticmethod
    def local_data_loading(file_path: str):
        if file_path.endswith('.pdf'):
            return load_file_loading.pdf_reader(file_path)
            
        elif file_path.endswith('.txt'):
            return load_file_loading.txt_reader(file_path)
        elif file_path.endswith('.csv'):
            pass
        else:
            return None
    
    @logger
    @staticmethod
    def load_hugginface_dataset(dataset_name : str ="rag-datasets/rag-mini-wikipedia" ):
        try:
            dataset = load_dataset(dataset_name)
            return dataset
        except Exception as e:
            print(f"Unknown exception occured {e}")
            raise
    

    @logger
    @staticmethod
    def load_local_data_folder(folder_path : str):
        folder_path = os.path.normpath(folder_path)  # Normalize folder path

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

        chunks = {}
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                file_path = os.path.normpath(file_path)  # Normalize path
                try:
                    data = Data_loading.local_data_loading(file_path)  # Pass full path to your function
                    chunks[file_path] = data  # Use full path as key
                except Exception as e:
                    print(f"Error loading file '{file_path}': {e}")
                    raise

        return chunks