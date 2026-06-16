# Báo cáo Phân tích Thất bại (Failure Analysis Report)

**Sinh viên:** Vũ Nhật Anh
**Lab:** Day 14 - AI Evaluation Factory

## 1. Tổng quan Benchmark
- **Tổng số cases:** 60 (Được tạo tự động bằng GPT-4o-mini SDG)
- **Điểm RAGAS trung bình (Giả lập):**
    - Faithfulness: 0.90
    - Relevancy: 0.80
    - Hit Rate: 1.00
    - MRR: 0.50
- **Điểm LLM-Judge trung bình (Thực tế):** ~3.42 / 5.0 (Chấm bởi GPT-4o-mini & Llama-3.3-70b-versatile)
- **Tỉ lệ đồng thuận (Agreement Rate):** [Ghi kết quả từ terminal vào đây]
- **Quyết định Release:** ✅ CHẤP NHẬN (APPROVE) do Delta Score +0.04.

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng (Ước tính) | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| API Rate Limit (429) | Nhiều (Đã khắc phục một phần) | Sử dụng bản Free tier của Groq (giới hạn 30 req/min) gây từ chối dịch vụ khi chạy song song 60 cases. |
| Suy giảm chất lượng Retrieval | Toàn bộ các câu khó | Lỗi Jina API (403) và xung đột thư viện `protobuf` khiến Cross-Encoder nội bộ sụp đổ, hệ thống tự lùi về (fallback) phương pháp tìm kiếm cơ bản. |
| Điểm số bị phạt (Penalized Score) | [Để trống] | Do lùi về tìm kiếm cơ bản, context trả về bị nhiễu, khiến LLM sinh ra câu trả lời không bám sát hoàn toàn Ground Truth. |

## 3. Phân tích 5 Whys (Nguyên nhân gốc rễ cho mức điểm 3.42/5.0)

### Case #1: Hệ thống RAG chưa đạt điểm tối đa (Chỉ đạt 3.42 thay vì 5.0)
1. **Symptom:** Điểm đánh giá chéo giữa 2 LLM Judge chỉ dao động quanh mức 3-4 điểm thay vì tuyệt đối.
2. **Why 1:** Câu trả lời của Agent bị thiếu một số ý chính hoặc độ liên quan chưa cao so với đáp án kỳ vọng (Ground Truth).
3. **Why 2:** Agent (LLM sinh text) nhận được các đoạn Context chưa phải là tốt nhất/sát nhất từ cơ sở dữ liệu.
4. **Why 3:** Hệ thống sắp xếp lại (Reranking) bằng AI đã bị sập (Crash) toàn bộ trong quá trình chạy.
5. **Why 4:** API Jina AI bị lỗi 403 (Hết token/Sai Key), đồng thời mô hình Reranker cục bộ (BAAI/bge-reranker-base) bị văng lỗi thư viện `protobuf` (MessageFactory object has no attribute GetPrototype).
6. **Root Cause (Nguyên nhân gốc):** Xung đột môi trường thư viện Python (Dependency Hell) và thiếu cơ chế quản lý API Key an toàn, dẫn đến hệ thống mất đi "vũ khí" Reranking quan trọng nhất, làm suy giảm nghiêm trọng độ chính xác của tài liệu đầu vào.

## 4. Kế hoạch cải tiến (Action Plan)
- [x] **Đã làm:** Bổ sung cơ chế `Exponential Backoff Auto-Retry` vào `MultiModelJudge` để tự động chờ và gửi lại request khi Groq báo lỗi 429 Rate Limit.
- [ ] Cài đặt lại thư viện `protobuf` đúng phiên bản (`pip install "protobuf<4.0.0"`) hoặc cấu hình biến môi trường để sửa triệt để lỗi của Local Reranker.
- [ ] Cập nhật lại Jina API Key (Task 7) hoặc mua gói trả phí để sử dụng tính năng Reranking trên mây.
- [ ] Tích hợp API thực tế của RAGAS vào `ExpertEvaluator` (thay cho bản Mock) để đo đạc chỉ số Hit Rate/MRR thật thay vì hardcode.
