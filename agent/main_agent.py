import asyncio
from typing import List, Dict
import sys
import os
from dotenv import load_dotenv

# Load file .env ở thư mục root ngay lập tức để các file khác có thể dùng
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Đảm bảo có thể import từ src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.task10_generation import generate_with_citation

class MainAgent:
    """
    Agent RAG thực tế được chuyển từ Day 08 sang.
    """
    def __init__(self):
        self.name = "SupportAgent-v2-Day08-RAG"

    async def query(self, question: str) -> Dict:
        """
        Gọi RAG pipeline thật từ Day 08.
        """
        # generate_with_citation là hàm đồng bộ (synchronous)
        # Ta dùng asyncio.to_thread để chạy nó không làm block event loop của Evaluator
        result = await asyncio.to_thread(generate_with_citation, question)
        
        # Format lại kết quả cho phù hợp với yêu cầu của hệ thống đánh giá (Evaluator)
        # Hệ thống đánh giá thường cần danh sách các đoạn text (string) trong 'contexts'
        contexts = [chunk['content'] for chunk in result.get('sources', [])]
        
        return {
            "answer": result["answer"],
            "contexts": contexts,
            "metadata": {
                "model": "gpt-4o-mini",
                "retrieval_source": result.get("retrieval_source", "none"),
                "sources_raw": result.get('sources', [])
            }
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = MainAgent()
    async def test():
        resp = await agent.query("Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?")
        print(f"Answer:\n{resp['answer']}\n")
        print(f"Số lượng contexts tìm được: {len(resp['contexts'])}")
    asyncio.run(test())
