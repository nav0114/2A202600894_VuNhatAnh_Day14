import sys
from pathlib import Path
import numpy as np

# Kiểm tra và import các thư viện cần thiết
try:
    import pypdfium2 as pdfium
except ImportError:
    print("Thiếu thư viện pypdfium2. Vui lòng cài đặt: pip install pypdfium2")
    sys.exit(1)

try:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        from rapidocr import RapidOCR
except ImportError:
    print("Thiếu thư viện rapidocr. Vui lòng cài đặt: pip install rapidocr-onnxruntime")
    sys.exit(1)

# Định nghĩa đường dẫn file
pdf_path = Path("data/landing/legal/nghi-dinh-105-2021.pdf")
out_path = Path("data/standardized/legal/nghi-dinh-105-2021.md")

if not pdf_path.exists():
    print(f"Không tìm thấy file PDF tại: {pdf_path}")
    sys.exit(1)

print(f"Bắt đầu OCR file {pdf_path.name} bằng RapidOCR (Offline hoàn toàn)...")

try:
    # Khởi động công cụ OCR offline
    ocr_engine = RapidOCR()
    
    # Mở tài liệu PDF
    doc = pdfium.PdfDocument(pdf_path)
    num_pages = len(doc)
    print(f"Tổng số trang: {num_pages}")
    
    full_text_pages = []
    
    for i in range(num_pages):
        print(f"Đang xử lý trang {i + 1}/{num_pages}...")
        
        # Render trang PDF thành ảnh PIL với độ phân giải cao (scale=2) để nhận diện chữ chính xác
        page = doc[i]
        bitmap = page.render(scale=2)
        pil_img = bitmap.to_pil()
        
        # Chuyển đổi ảnh sang dạng numpy array để đưa vào OCR
        img_np = np.array(pil_img)
        
        # Nhận diện chữ trên trang (trả về đối tượng RapidOCROutput)
        output = ocr_engine(img_np)
        
        page_text = ""
        if output is not None:
            # Trường hợp 1: Đối tượng RapidOCROutput (phiên bản mới)
            if hasattr(output, "txts") and output.txts:
                page_text = "\n".join(output.txts)
            
            # Trường hợp 2: Tuple dạng (result, elapse) (phiên bản cũ)
            elif isinstance(output, tuple) and len(output) >= 1:
                result = output[0]
                if result:
                    page_lines = [line[1] for line in result]
                    page_text = "\n".join(page_lines)
            
            # Trường hợp 3: List kết quả trực tiếp
            elif isinstance(output, list):
                page_lines = [line[1] for line in output if len(line) > 1]
                page_text = "\n".join(page_lines)
                
        if page_text:
            full_text_pages.append(f"## Trang {i + 1}\n\n{page_text}")
        else:
            full_text_pages.append(f"## Trang {i + 1}\n\n[Trang rỗng / Không nhận diện được chữ]")
            
    # Ghép nội dung toàn bộ các trang lại với nhau
    full_content = "\n\n---\n\n".join(full_text_pages)
    
    # Ghi dữ liệu ra file Markdown
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_content, encoding="utf-8")
    
    print(f"\n✓ Hoàn thành! File đã được lưu tại: {out_path}")
    print(f"Độ dài văn bản trích xuất: {len(full_content)} ký tự.")
    
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")