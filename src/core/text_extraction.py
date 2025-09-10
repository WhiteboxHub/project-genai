# text_extraction.py
import csv
import PyPDF2
import os

class text_extraction:

    # extract text from Pdf
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        reader = PyPDF2.PdfReader(pdf_path)
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
                text.append(" ".join(row))
        return "\n".join(text)

# Individual functions for each file type
def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    return text_extraction.extract_text_from_pdf(pdf_path)

def extract_txt_text(txt_path):
    """Extract text from TXT file"""
    return text_extraction.extract_text_from_txt(txt_path)

def extract_csv_text(csv_path):
    """Extract text from CSV file"""
    return text_extraction.extract_text_from_csv(csv_path)

# Main function for easy import
def text_extraction_main(file_path):
    """
    Main function to extract text from PDF, CSV, and TXT files
    """
    if file_path.endswith('.pdf'):
        return extract_pdf_text(file_path)
    elif file_path.endswith('.txt'):
        return extract_txt_text(file_path)
    elif file_path.endswith('.csv'):
        return extract_csv_text(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF, TXT, and CSV are supported")

if __name__ == "__main__":  
    # Your specific file path
    txt_file_path = r"C:\Users\aknar\OneDrive\Documents\Desktop\ai-agent\project-genai\Data\sample.txt"
    
    # Test with your specific text file
    try:
        print(f"\n=== Testing TXT file ===")
        print(f"File path: {txt_file_path}")
        
        # Check if file exists
        if os.path.exists(txt_file_path):
            result = text_extraction_main(txt_file_path)
            print(f"\nExtracted text:")
            print("=" * 50)
            print(result)
            print("=" * 50)
            print(f"\nTotal characters: {len(result)}")
        else:
            print(f"Error: File not found at {txt_file_path}")
            print("Please check the file path and try again.")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # add files later
    # pdf_file_path = r"C:\Users\aknar\OneDrive\Documents\Desktop\ai-agent\project-genai\Data\sample.pdf"
    # csv_file_path = r"C:\Users\aknar\OneDrive\Documents\Desktop\ai-agent\project-genai\Data\sample.csv
    #PS C:\Users\aknar\OneDrive\Documents\Desktop\ai-agent\project-genai> & C:/Users/aknar/AppData/Local/Programs/Python/Python39/python.exe c:/Users/aknar/OneDrive/Documents/Desktop/ai-agent/project-genai/src/core/text_extraction.py

"""
output
Extracted text:
==================================================
Cardiovascular diseases are a group of disorders affecting the heart and blood vessels. They include conditions such as coronary artery disease, heart failure, arrhythmias, and hypertension. Early detection and lifestyle modifications are crucial for managing these diseases effectively.

Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels. It can lead to complications such as neuropathy, retinopathy, kidney disease, and cardiovascular problems. Proper management includes monitoring blood glucose, following a healthy diet, and taking medications as prescribed.

Respiratory diseases, such as asthma and chronic obstructive pulmonary disease (COPD), affect the lungs and breathing capacity. These conditions can be triggered by environmental factors, genetics, or infections. Treatments may include inhalers, medications, and lifestyle adjustments to reduce symptoms.

Medical imaging technologies, such as MRI, CT scans, and X-rays, play a vital role in diagnosis and treatment planning. They allow clinicians to visualize internal organs and detect abnormalities with high precision.

Vaccinations are essential for preventing infectious diseases. Immunization programs help reduce the prevalence of illnesses such as influenza, measles, and hepatitis. Public awareness and accessibility to vaccines are critical to ensure widespread protection.

Emerging medical research continues to focus on personalized medicine, which tailors treatments based on an individual’s genetic profile, lifestyle, and environment. This approach aims to improve outcomes and reduce adverse effects.
==================================================
"""