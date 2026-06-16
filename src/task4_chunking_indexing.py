"""
Task 4 — Chunking & Indexing vào Vector Store sử dụng OpenAI Embedding.
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load env variables
load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Cấu hình tham số RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Cấu hình OpenAI Embedding Model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
VECTOR_STORE = "local_json"

# Khởi tạo OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "Chưa tìm thấy OPENAI_API_KEY trong file .env hoặc biến môi trường. "
        "Hãy điền OPENAI_API_KEY vào file .env ở thư mục gốc."
    )

client = OpenAI(api_key=api_key)


def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia nhỏ văn bản bằng cơ chế dịch chuyển cửa sổ (sliding window)."""
    chunks = []
    for doc in documents:
        content = doc["content"]
        start = 0
        i = 0
        while start < len(content):
            end = start + CHUNK_SIZE
            chunk_text = content[start:end]
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
            start += CHUNK_SIZE - CHUNK_OVERLAP
            i += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Nhúng vector sử dụng API OpenAI text-embedding-3-small."""
    import time
    texts = [c["content"] for c in chunks]
    print(f"Đang tiến hành nhúng {len(texts)} chunks bằng OpenAI API...")
    
    # OpenAI hỗ trợ batch size lớn hơn (tối đa 2048)
    batch_size = 100
    embeddings = []
    
    for idx in range(0, len(texts), batch_size):
        batch_texts = texts[idx:idx+batch_size]
        
        # Thử gọi API kèm cơ chế tự động thử lại khi bị giới hạn tần suất (Rate Limit / 429)
        for attempt in range(5):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch_texts
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if "429" in str(e) or "RateLimitError" in e.__class__.__name__:
                    print(f"  [Rate Limit] Đang chờ 10 giây trước khi thử lại (Lần thử {attempt+1}/5)...")
                    time.sleep(10.0)
                else:
                    raise e
        
        print(f"  Đã nhúng: {min(idx+batch_size, len(texts))}/{len(texts)} chunks...")
        time.sleep(0.5)  # Tránh spam quá nhanh
        
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
        
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu trữ kết quả nhúng vào file JSON cục bộ."""
    vector_store_path = Path(__file__).parent.parent / "data" / "vector_store.json"
    vector_store_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    print(f"✓ Đã lưu Vector Store cục bộ tại: {vector_store_path}")


def run_pipeline():
    print("=" * 50)
    print("Task 4: Chunking & Indexing với OpenAI API")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
