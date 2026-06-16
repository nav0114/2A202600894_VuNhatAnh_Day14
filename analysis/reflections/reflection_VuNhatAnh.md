# Báo cáo Cá nhân (Individual Reflection)

**Họ và tên:** Vũ Nhật Anh
**Lab:** Day 14 - AI Evaluation Factory

## 1. Bài học rút ra (Key Learnings)
- Hiểu được tầm quan trọng của việc xây dựng "Golden Dataset" để đánh giá định lượng cho một hệ thống RAG thay vì chỉ kiểm tra "bằng mắt" một vài câu hỏi ngẫu nhiên.
- Biết cách sử dụng kĩ thuật SDG (Synthetic Data Generation) để tự động hóa việc tạo ra hàng chục câu hỏi test phức tạp từ dữ liệu thô.
- Cấu hình và triển khai thành công mô hình "Multi-Judge" (Dùng nhiều LLM như GPT-4o-mini và Llama-3.3-70b để đóng vai ban giám khảo), từ đó thấy rõ được độ chênh lệch và cách xử lý xung đột điểm số giữa các AI.
- Xử lý được các lỗi thực tế liên quan đến giới hạn API (Rate Limit 429) bằng cơ chế Auto-Retry.

## 2. Khó khăn gặp phải (Challenges)
- Gặp lỗi tương thích thư viện (`protobuf`) làm sập hệ thống Local Reranker.
- Lỗi giới hạn tốc độ (Rate limit) khi gọi API của Groq miễn phí lúc chạy song song quá nhiều câu hỏi cùng lúc.

## 3. Đề xuất cải tiến (Proposed Improvements)
- Tối ưu hóa lại kiến trúc `async` để kiểm soát tốt hơn số lượng request đồng thời (Concurrency Control), tránh tình trạng spam API dẫn đến bị chặn.
- Nâng cấp phần RAGAS (ExpertEvaluator) từ giả lập sang tích hợp thật để đánh giá chi tiết chỉ số Hit Rate & MRR của Vector Database.
