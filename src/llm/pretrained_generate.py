"""Answer generation using a pretrained instruction-tuned LLM (Qwen2.5-0.5B-Instruct).

Provides direct, concise answers grounded in retrieved context chunks.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class PretrainedGenerator:
    """Wrapper around a pretrained instruct model for RAG answer generation."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        max_new_tokens: int = 200,
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = self._select_device(device)

        LOGGER.info("Loading pretrained LLM: %s on %s (local_files_only=%s)",
                    model_name, self.device, local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        LOGGER.info("Pretrained LLM loaded successfully.")

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested != "auto":
            return torch.device(requested)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def generate_answer(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        *,
        max_context_chars: int = 4000,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate a direct answer from question + retrieved chunks."""
        context = self._build_context(chunks, max_context_chars)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions about Indian government policy documents. "
                    "Rules:\n"
                    "1. Answer ONLY using the provided context. Do not add information not present in the context.\n"
                    "2. Be direct and concise - start with the specific answer, then add 1-2 sentences of supporting detail.\n"
                    "3. If the context contains specific numbers, dates, or names, quote them exactly.\n"
                    "4. Keep your answer under 100 words.\n"
                    "5. If the answer is not in the context, say \"Not found in the provided documents.\""
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = max_new_tokens or self.max_new_tokens

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        # Decode only the generated part (skip the prompt)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]], max_chars: int) -> str:
        parts: list[str] = []
        total = 0
        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(text[:remaining])
                break
            parts.append(text)
            total += len(text)
        return "\n\n".join(parts)
