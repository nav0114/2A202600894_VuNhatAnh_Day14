"""
Task 5 — Semantic Search Module sử dụng OpenAI Embedding.
"""

import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Cấu hình API Key cho OpenAI
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm ngữ nghĩa sử dụng vector similarity từ OpenAI."""
    vector_store_path = Path(__file__).parent.parent / "data" / "vector_store.json"
    if not vector_store_path.exists():
        print("Không tìm thấy Vector Store. Hãy chạy Task 4 trước.")
        return []

    with open(vector_store_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        return []

    # Kiểm tra API key lúc truy vấn
    current_key = os.environ.get("OPENAI_API_KEY")
    if not current_key:
        print("Thiếu OPENAI_API_KEY trong file .env hoặc biến môi trường.")
        return []
    
    global client
    client = OpenAI(api_key=current_key)

    # Embed câu truy vấn (query) bằng OpenAI API (mô hình text-embedding-3-small)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = np.array(response.data[0].embedding)

    results = []
    for chunk in chunks:
        emb = np.array(chunk["embedding"])
        # Tính cosine similarity
        dot_product = np.dot(query_embedding, emb)
        norm_q = np.linalg.norm(query_embedding)
        norm_e = np.linalg.norm(emb)
        similarity = dot_product / (norm_q * norm_e) if norm_q > 0 and norm_e > 0 else 0.0

        results.append({
            "content": chunk["content"],
            "score": float(similarity),
            "metadata": chunk["metadata"]
        })

    # Sắp xếp giảm dần theo score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Chạy thử truy vấn tìm kiếm
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
