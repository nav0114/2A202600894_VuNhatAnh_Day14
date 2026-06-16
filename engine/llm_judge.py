import asyncio
from typing import Dict, Any
import os
from openai import AsyncOpenAI

class MultiModelJudge:
    def __init__(self):
        # Khởi tạo client cho OpenAI (GPT)
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Khởi tạo client cho Groq (Dùng chung bộ OpenAI SDK được vì Groq tương thích)
        # Biến môi trường user gõ là GROD thay vì GROQ, ta lấy nguyên văn
        self.groq_client = AsyncOpenAI(
            api_key=os.getenv("GROQ_API_KEY"), 
            base_url="https://api.groq.com/openai/v1"
        )

    async def _get_score(self, client: AsyncOpenAI, model: str, question: str, answer: str, ground_truth: str, max_retries: int = 3) -> float:
        prompt = f"""You are an expert AI evaluator.
Please evaluate the Answer based on the Question and the Ground Truth.
Give a score from 1 to 5 (1 being completely wrong, 5 being perfectly correct and matching the ground truth).
Return ONLY the numeric score (1, 2, 3, 4, or 5). Do not output anything else.

Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}"""
        
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                score_text = response.choices[0].message.content.strip()
                # Lọc lấy chữ số đầu tiên làm điểm (tránh model nói lan man)
                score = float(''.join(filter(str.isdigit, score_text))[0]) if any(c.isdigit() for c in score_text) else 1.0
                return min(max(score, 1.0), 5.0)
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg:
                    wait_time = (attempt + 1) * 3
                    print(f"⏳ Bị giới hạn tốc độ (Rate Limit) cho model {model}. Đang chờ {wait_time}s để thử lại...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"⚠ Lỗi khi gọi model {model}: {e}")
                    return 1.0  # Lỗi khác thì trả về 1 điểm luôn
        return 1.0 # Nếu thử quá số lần vẫn lỗi thì đành chịu

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Gọi 2 model (GPT-4o-mini và Llama-3.3-70b-versatile) chấm điểm song song
        """
        # Chạy 2 task gọi API cùng lúc cho nhanh
        task_gpt = self._get_score(self.openai_client, "gpt-4o-mini", question, answer, ground_truth)
        task_llama = self._get_score(self.groq_client, "llama-3.3-70b-versatile", question, answer, ground_truth)
        
        score_gpt, score_llama = await asyncio.gather(task_gpt, task_llama)
        
        avg_score = (score_gpt + score_llama) / 2
        
        # Tính độ đồng thuận (Agreement Rate)
        diff = abs(score_gpt - score_llama)
        if diff == 0:
            agreement = 1.0
        elif diff <= 1:
            agreement = 0.5
        else:
            agreement = 0.0
            
        reasoning = f"GPT: {score_gpt}/5, Llama: {score_llama}/5. "
        if diff == 0:
            reasoning += "Hai giám khảo hoàn toàn đồng thuận."
        elif diff <= 1:
            reasoning += "Hai giám khảo có chút khác biệt nhỏ."
        else:
            reasoning += "XUNG ĐỘT! Hai giám khảo không đồng ý với nhau."
            
        return {
            "final_score": avg_score,
            "agreement_rate": agreement,
            "individual_scores": {"gpt-4o-mini": score_gpt, "llama-3.3-70b": score_llama},
            "reasoning": reasoning
        }
