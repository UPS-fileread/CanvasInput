import fitz
import os

def pdf_to_text(pdf_bytes, num_pages=5):
    """
    Convert the first num_pages of a PDF (from bytes) to text.
    Returns the extracted text as a string.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages_to_use = doc[:num_pages]
        text = "\n\n".join(page.get_text() for page in pages_to_use)
    return text

def extract_first_n_pages(pdf_bytes, n=5):
    """
    Return a bytes object containing a new PDF with the first n pages.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        new_pdf = fitz.open()
        for i in range(min(n, len(doc))):
            new_pdf.insert_pdf(doc, from_page=i, to_page=i)
        return new_pdf.tobytes()