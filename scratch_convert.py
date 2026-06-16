import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import sys
from pathlib import Path
from docling.document_converter import DocumentConverter

pdf_path = Path("/Users/HELLO/Project/lab-vinAI/2A202600894_VuNhatAnh_Day08/data/landing/legal/nghi-dinh-105-2021.pdf")
out_path = Path("/Users/HELLO/Project/lab-vinAI/2A202600894_VuNhatAnh_Day08/data/standardized/legal/nghi-dinh-105-2021.md")

print(f"Converting {pdf_path} to {out_path} using Docling (with hf-mirror and disabled SSL checks)...")
converter = DocumentConverter()
result = converter.convert(pdf_path)
markdown_content = result.document.export_to_markdown()

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(markdown_content, encoding="utf-8")
print("Conversion complete!")
