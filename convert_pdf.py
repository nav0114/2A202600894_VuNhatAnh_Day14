import os

# 1. Cấu hình biến môi trường để tăng tốc tải mô hình qua Mirror và tránh lỗi chặn kết nối
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. Vô hiệu hóa kiểm tra SSL của Hugging Face (phòng trường hợp môi trường Conda của bạn bị lỗi chứng chỉ SSL)
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pathlib import Path
from docling.document_converter import DocumentConverter

# Đường dẫn file nguồn và file đích
pdf_path = Path("data/landing/legal/nghi-dinh-105-2021.pdf")
out_path = Path("data/standardized/legal/nghi-dinh-105-2021.md")

print(f"Đang tiến hành chuyển đổi {pdf_path.name} sang Markdown sử dụng Docling...")

try:
    # Khởi tạo bộ chuyển đổi Docling
    converter = DocumentConverter()
    
    # Thực hiện chuyển đổi (sẽ tự động tải các mô hình layout/OCR cần thiết)
    result = converter.convert(pdf_path)
    
    # Xuất tài liệu ra định dạng Markdown
    markdown_content = result.document.export_to_markdown()
    
    # Tạo thư mục đích nếu chưa có và lưu nội dung
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown_content, encoding="utf-8")
    
    print(f"✓ Chuyển đổi thành công! Nội dung đã được ghi vào: {out_path}")
except Exception as e:
    print(f"Có lỗi xảy ra trong quá trình chuyển đổi: {e}")