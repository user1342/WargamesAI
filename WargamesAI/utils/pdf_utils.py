from fpdf import FPDF
import PyPDF2
import hashlib

def write_pdf(pdf_data_str, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, pdf_data_str)
    pdf.output(pdf_path)
    return True 

def read_pdf(pdf_path):
    data = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        number_of_pages = len(reader.pages)
        
        
        # Extract text from each page
        for page_num in range(number_of_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            data.append(text)

    return " ".join(text)

def hash_string(input_string, algorithm='sha256'):
    """
    Hashes a string using the specified algorithm.

    Parameters:
    - input_string (str): The string to be hashed.
    - algorithm (str): The hashing algorithm to use (e.g., 'sha256', 'md5', 'sha1').

    Returns:
    - str: The resulting hash in hexadecimal format.
    """
    input_bytes = input_string.encode('utf-8')
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(input_bytes)
    return hash_obj.hexdigest()