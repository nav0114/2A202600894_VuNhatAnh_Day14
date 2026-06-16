import pdfplumber
from pathlib import Path

pdf_path = Path("/Users/HELLO/Project/lab-vinAI/2A202600894_VuNhatAnh_Day08/data/landing/legal/nghi-dinh-105-2021.pdf")
print("Extracting with pdfplumber...")
try:
    with pdfplumber.open(pdf_path) as pdf:
        text = []
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
            print(f"Page {i+1} extracted: {len(page_text) if page_text else 0} chars")
        full_text = "\n\n".join(text)
        print(f"Total extracted: {len(full_text)} chars")
except Exception as e:
    print(f"pdfplumber error: {e}")
