import json
import asyncio
import os
import random
from typing import List, Dict
from dotenv import load_dotenv

# Load file .env ở thư mục root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from openai import AsyncOpenAI

async def generate_qa_from_text(client: AsyncOpenAI, text: str, num_pairs: int = 3) -> List[Dict]:
    """
    Sử dụng OpenAI API để tạo các cặp (Question, Expected Answer, Context).
    """
    prompt = f"""Dựa vào đoạn văn bản sau, hãy tạo {num_pairs} câu hỏi trắc nghiệm tự luận (QA pairs) để kiểm tra kiến thức.
Yêu cầu:
1. Các câu hỏi phải xoay quanh các ý chính trong văn bản.
2. Phải có ít nhất 1 câu hỏi "lừa" (hỏi một chi tiết không có hoặc sai lệch so với văn bản, và câu trả lời kỳ vọng phải chỉ ra là văn bản không đề cập hoặc phủ nhận điều đó).
3. Câu trả lời kỳ vọng (expected_answer) phải chính xác theo văn bản.

Văn bản:
{text}

Hãy trả về CHỈ MỘT MẢNG JSON hợp lệ theo định dạng sau (không markdown, không bọc bằng ```json):
[
  {{
    "question": "Câu hỏi...",
    "expected_answer": "Câu trả lời mẫu...",
    "context": "Trích đoạn nhỏ từ văn bản chứa câu trả lời...",
    "metadata": {{"difficulty": "easy/hard/adversarial", "type": "fact-check"}}
  }}
]
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        
        content = response.choices[0].message.content.strip()
        # Clean up markdown if any
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        qa_pairs = json.loads(content.strip())
        
        # Override context to be the full text chunk if the LLM abbreviated it too much
        for pair in qa_pairs:
            pair["context"] = text
            
        return qa_pairs
    except Exception as e:
        print(f"Lỗi khi generate QA: {e}")
        return []

async def main():
    print("🚀 Bắt đầu tạo Golden Dataset (SDG)...")
    
    # 1. Đọc dữ liệu từ vector_store.json
    vector_store_path = "data/vector_store.json"
    if not os.path.exists(vector_store_path):
        print(f"❌ Không tìm thấy {vector_store_path}")
        return
        
    with open(vector_store_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"✅ Đã tải {len(data)} chunks từ vector store.")
    
    # Lấy ngẫu nhiên khoảng 20 chunks để sinh ra ~60 câu hỏi
    random.seed(42)  # Để reproducible
    sampled_chunks = random.sample(data, min(20, len(data)))
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    all_qa_pairs = []
    
    # Chạy song song (batch) để nhanh hơn
    print(f"⏳ Đang gửi {len(sampled_chunks)} request đến GPT-4o-mini để sinh câu hỏi...")
    tasks = [generate_qa_from_text(client, chunk["content"], 3) for chunk in sampled_chunks]
    results = await asyncio.gather(*tasks)
    
    for qa_list in results:
        all_qa_pairs.extend(qa_list)
        
    print(f"✅ Đã tạo thành công {len(all_qa_pairs)} test cases.")
    
    # Lưu vào file
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in all_qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print("🎉 Done! Saved to data/golden_set.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
