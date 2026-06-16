"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # Tách thành phần tử chỉ số chẵn (0, 2, 4...) đặt ở đầu
    # Và phần tử chỉ số lẻ (1, 3, 5...) đảo ngược lại đặt ở cuối
    evens = [chunks[i] for i in range(len(chunks)) if i % 2 == 0]
    odds = [chunks[i] for i in range(len(chunks)) if i % 2 != 0]
    odds.reverse()
    return evens + odds


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call LLM
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


def condense_query(query: str, history: list[dict]) -> str:
    """
    Sử dụng Gemini để chuyển đổi câu hỏi follow-up và chat history thành câu hỏi độc lập (standalone query) để retrieval tốt hơn.
    """
    if not history:
        return query

    import google.generativeai as genai
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key or gemini_api_key == "xxx":
        return query

    genai.configure(api_key=gemini_api_key)
    
    # Sử dụng model gemini-3.1-flash-lite
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    
    # Xây dựng prompt
    history_str = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
        
    prompt = f"""Dưới đây là lịch sử hội thoại giữa Người dùng và Trợ lý ảo, cùng với một câu hỏi tiếp theo của Người dùng.
Hãy chuyển câu hỏi tiếp theo này thành một câu hỏi độc lập duy nhất (standalone question) bằng tiếng Việt để có thể dùng tìm kiếm tài liệu pháp luật.
Không trả lời câu hỏi, chỉ trả về duy nhất câu hỏi độc lập được viết lại.

Lịch sử hội thoại:
{history_str}

Câu hỏi tiếp theo: {query}

Standalone Question:"""

    try:
        response = model.generate_content(prompt)
        standalone = response.text.strip()
        if standalone:
            return standalone
    except Exception as e:
        print(f"Error in condense_query: {e}")
        
    return query


def generate_with_citation_and_memory(query: str, history: list[dict], top_k: int = TOP_K) -> dict:
    """
    RAG Generation có citation và hỗ trợ hội thoại (conversation memory).
    """
    # 1. Condense query để tìm kiếm chính xác hơn
    standalone_query = condense_query(query, history)
    
    # 2. Retrieve using the standalone query
    chunks = retrieve(standalone_query, top_k=top_k)
    
    # 3. Reorder
    reordered = reorder_for_llm(chunks)
    
    # 4. Format context
    context = format_context(reordered)
    
    # 5. Call LLM with History
    import google.generativeai as genai
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key or gemini_api_key == "xxx":
        raise ValueError("Hãy cấu hình GEMINI_API_KEY hợp lệ trong file .env")

    genai.configure(api_key=gemini_api_key)
    
    model = genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite",
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
    )
    
    # Xây dựng prompt chứa cả lịch sử và context mới
    prompt_parts = []
    if history:
        prompt_parts.append("Lịch sử hội thoại trước đó:")
        for msg in history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            prompt_parts.append(f"{role}: {msg['content']}")
        prompt_parts.append("\n---\n")
        
    prompt_parts.append(f"Tài liệu tham khảo mới nhất (Context):\n{context}")
    prompt_parts.append(f"\n---\nCâu hỏi hiện tại của Người dùng: {query}")
    
    user_message = "\n".join(prompt_parts)
    
    response = model.generate_content(user_message)
    answer = response.text
    
    retrieval_source = "none"
    if chunks:
        retrieval_source = chunks[0].get("source", "hybrid")
        
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
        "standalone_query": standalone_query
    }


def generate_with_citation_and_memory_stream(query: str, history: list[dict], top_k: int = TOP_K):
    """
    RAG Generation streaming version for streaming response in UI.
    Yields:
        - {"type": "metadata", "sources": list[dict], "retrieval_source": str, "standalone_query": str} (first item)
        - {"type": "content", "delta": str} (subsequent items)
    """
    import os
    
    # 1. Condense query
    standalone_query = condense_query(query, history)
    
    # 2. Retrieve
    chunks = retrieve(standalone_query, top_k=top_k)
    
    # 3. Reorder & format context
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    
    retrieval_source = "none"
    if chunks:
        retrieval_source = chunks[0].get("source", "hybrid")
        
    # Yield metadata first (so UI can render stats instantly)
    yield {
        "type": "metadata",
        "sources": chunks,
        "retrieval_source": retrieval_source,
        "standalone_query": standalone_query
    }
    
    # 4. Call LLM with streaming
    import google.generativeai as genai
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key or gemini_api_key == "xxx":
        raise ValueError("Hãy cấu hình GEMINI_API_KEY hợp lệ trong file .env")

    genai.configure(api_key=gemini_api_key)
    
    model = genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite",
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
    )
    
    prompt_parts = []
    if history:
        prompt_parts.append("Lịch sử hội thoại trước đó:")
        for msg in history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            prompt_parts.append(f"{role}: {msg['content']}")
        prompt_parts.append("\n---\n")
        
    prompt_parts.append(f"Tài liệu tham khảo mới nhất (Context):\n{context}")
    prompt_parts.append(f"\n---\nCâu hỏi hiện tại của Người dùng: {query}")
    
    user_message = "\n".join(prompt_parts)
    
    response = model.generate_content(user_message, stream=True)
    for chunk in response:
        if chunk.text:
            yield {
                "type": "content",
                "delta": chunk.text
            }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
