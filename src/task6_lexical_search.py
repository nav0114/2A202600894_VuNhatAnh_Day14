"""
Task 6 — Lexical Search Module (BM25).
"""

import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

# Đường dẫn đến file vector store để lấy corpus đồng bộ
VECTOR_STORE_PATH = Path(__file__).parent.parent / "data" / "vector_store.json"

CORPUS: list[dict] = []
bm25_index = None


def load_corpus():
    """Tải dữ liệu corpus và khởi tạo index BM25."""
    global CORPUS, bm25_index
    if not CORPUS:
        if VECTOR_STORE_PATH.exists():
            with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
                CORPUS = json.load(f)
        else:
            # Fallback nếu chưa chạy Task 4
            from src.task4_chunking_indexing import load_documents, chunk_documents
            docs = load_documents()
            CORPUS = chunk_documents(docs)
            
        if CORPUS:
            bm25_index = build_bm25_index(CORPUS)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # Tokenize đơn giản bằng cách tách từ khoảng trắng và chuyển chữ thường
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    load_corpus()
    if not CORPUS or bm25_index is None:
        return []

    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    # Sắp xếp các chỉ số của corpus theo score giảm dần
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        # Chỉ lấy các kết quả có score > 0 (có khớp từ khóa)
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test thử tìm kiếm từ khóa
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
