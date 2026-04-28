"""Cross-encoder reranker for hybrid retrieval candidates."""

from __future__ import annotations

import logging
import signal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder

LOGGER = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-rank retrieval candidates with a cross-encoder."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        batch_size: int = 32,
        device: str = "cpu",
        local_files_only: bool = False,
        model_load_timeout_s: int = 60,
    ) -> None:
        """Load cross-encoder model with optional local-only mode."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.local_files_only = local_files_only
        self.model_load_timeout_s = model_load_timeout_s

        LOGGER.info(
            "Loading reranker model: %s (device=%s, local_files_only=%s)",
            self.model_name,
            self.device,
            self.local_files_only,
        )
        self.model = self._load_model()

    def _load_model(self) -> CrossEncoder:
        """Load model with fallback to local cache when network fails."""
        from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder

        def _timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(
                f"Timed out after {self.model_load_timeout_s}s while loading reranker model "
                f"'{self.model_name}'."
            )

        import threading
        use_timeout = (
            hasattr(signal, "SIGALRM")
            and self.model_load_timeout_s > 0
            and threading.current_thread() is threading.main_thread()
        )
        previous_handler = None
        if use_timeout:
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.model_load_timeout_s)

        try:
            if self.local_files_only:
                return CrossEncoder(self.model_name, device=self.device, local_files_only=True)

            try:
                return CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:
                LOGGER.warning(
                    "Online reranker model load failed for %s; retrying with local cache only. Reason: %s",
                    self.model_name,
                    exc,
                )
                try:
                    return CrossEncoder(self.model_name, device=self.device, local_files_only=True)
                except Exception as local_exc:
                    raise RuntimeError(
                        f"Failed to load reranker model '{self.model_name}'. "
                        "Download it once with network access or pass a local model path."
                    ) from local_exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"{exc} This environment is likely stuck importing transformers/sklearn. "
                "Try a Python 3.11 venv for stable startup."
            ) from exc
        finally:
            if use_timeout:
                signal.alarm(0)
                if previous_handler is not None:
                    signal.signal(signal.SIGALRM, previous_handler)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Re-rank candidates and append `rerank_score`."""
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if not candidates:
            return []

        pairs: list[tuple[str, str]] = []
        for candidate in candidates:
            pairs.append((query, str(candidate.get("text", ""))))

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
        return reranked[:top_k]
