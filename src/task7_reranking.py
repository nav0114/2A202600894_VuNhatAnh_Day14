"""
Task 7 — Reranking Module.
"""

import os
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

# Load env variables
load_dotenv()

JINA_API_KEY = os.environ.get("JINA_API_KEY")


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Tính cosine similarity giữa hai vector."""
    vec_a = np.array(a)
    vec_b = np.array(b)
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return float(dot / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Jina Reranker API hoặc local BAAI/bge-reranker-base).
    """
    if not candidates:
        return []

    # 1. Thử dùng Jina Reranker API nếu có key hợp lệ
    if JINA_API_KEY and JINA_API_KEY != "jina_xxx" and JINA_API_KEY.strip() != "":
        import requests
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=15
            )
            if response.status_code == 200:
                reranked = response.json()["results"]
                return [
                    {**candidates[r["index"]], "score": float(r["relevance_score"])}
                    for r in reranked
                ]
            else:
                print(f"Jina API returned status {response.status_code}, falling back to local reranker.")
        except Exception as e:
            print(f"Jina API error: {e}, falling back to local reranker.")

    # 2. Fallback: Sử dụng model BAAI/bge-reranker-base chạy offline bằng transformers & torch
    print("Đang chạy Cross-Encoder cục bộ (BAAI/bge-reranker-base)...")
    try:
        model_name = "BAAI/bge-reranker-base"
        # Sử dụng local_files_only=True để tránh treo terminal khi tải model từ Hugging Face nếu mạng bị chặn
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        model.eval()

        pairs = [[query, c["content"]] for c in candidates]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors='pt')
            scores = model(**inputs).logits.view(-1).float()
            # Áp dụng sigmoid để đưa về khoảng [0, 1]
            probs = torch.sigmoid(scores).cpu().numpy().tolist()

        # Cập nhật điểm và sắp xếp
        scored_candidates = []
        for c, prob in zip(candidates, probs):
            item = c.copy()
            item["score"] = float(prob)
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]

    except Exception as e:
        print(f"Không thể chạy local cross-encoder: {e}. Sử dụng fallback sắp xếp cơ bản.")
        # Fallback tối giản nếu không tải được model: giữ nguyên thứ tự sắp xếp và giới hạn top_k
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
        return sorted_candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    if not candidates:
        return []

    # Kiểm tra xem các candidates có chứa vector embedding không
    # Nếu không có, ta sử dụng cosine similarity bằng cách sinh lại (hoặc bỏ qua)
    # Ở đây giả định candidates đã có 'embedding' từ khâu retrieval
    for c in candidates:
        if "embedding" not in c:
            raise ValueError("Candidates must contain 'embedding' field for MMR reranking.")

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # relevance: độ tương đồng với query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # max_sim_to_selected: độ tương đồng lớn nhất với các tài liệu đã chọn trước đó
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # Công thức tính điểm MMR
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker khác nhau (dense và lexical).
    RRF(d) = Σ 1 / (k + rank_r(d))
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            # Tính tổng điểm RRF của tài liệu qua các bảng xếp hạng
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    # Sắp xếp giảm dần theo điểm RRF
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # MMR cần query_embedding, ta có thể sinh qua API của OpenAI
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Need OPENAI_API_KEY in env for MMR query embedding.")
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = response.data[0].embedding
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        # RRF cần nhiều ranked lists (ví dụ: [[dense_results], [lexical_results]])
        # Trong interface này, ta chỉ gộp danh sách ban đầu làm 1 danh sách
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Chạy thử với dữ liệu giả lập
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
