from PyPDF2 import PdfReader
import os

def extract_text_from_pdf(pdf_path,chunk_size=500) -> str:
    reader=PdfReader(pdf_path)
    full_text=""
    for page in reader.pages:
        full_text+=page.extract_text() or ""
    return [full_text[i:i+chunk_size] for i in range(0,len(full_text),chunk_size)]
        
    
