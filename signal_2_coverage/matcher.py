"""
Signal 2: Dense Semantic Verification & Alignment Matcher (BGE-M3 / MiniLM)
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer

from .config import CoverageConfig


class SemanticCoverageMatcher:
    """
    Computes an N x M all-to-all dense semantic alignment matrix between
    Citizen Query items (Q_1 ... Q_N) and PIO Response statements (R_1 ... R_M).
    Identifies maximum semantic similarity regardless of statement order.
    """
    def __init__(self, config: Optional[CoverageConfig] = None):
        self.config = config or CoverageConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_name = None
        self._load_embedding_model()

    def _load_embedding_model(self):
        """
        Loads local cached embedding models first (BGE-M3 or MiniLM) before attempting network downloads.
        """
        # 1. Try local primary model
        try:
            self.model = SentenceTransformer(self.config.primary_embedding_model, device=self.device, local_files_only=True)
            self.model_name = self.config.primary_embedding_model
            print(f"[Signal 2 Matcher] Loaded local primary model: {self.model_name}")
            return
        except Exception:
            pass

        # 2. Try local fallback model (e.g. all-MiniLM-L6-v2)
        try:
            self.model = SentenceTransformer(self.config.fallback_embedding_model, device=self.device, local_files_only=True)
            self.model_name = self.config.fallback_embedding_model
            print(f"[Signal 2 Matcher] Loaded local fallback model: {self.model_name}")
            return
        except Exception:
            pass

        # 3. If neither is available locally, attempt network load
        try:
            print(f"[Signal 2 Matcher] Local models not found. Downloading fallback model: {self.config.fallback_embedding_model}...")
            self.model = SentenceTransformer(self.config.fallback_embedding_model, device=self.device)
            self.model_name = self.config.fallback_embedding_model
            print(f"[Signal 2 Matcher] Successfully downloaded and loaded: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load any embedding model: {e}")

    def compute_alignment_matrix(self, questions: List[str], responses: List[str]) -> np.ndarray:
        """
        Builds the N x M normalized cosine similarity matrix.
        Shape: [len(questions), len(responses)]
        """
        if not questions or not responses:
            return np.zeros((len(questions), len(responses)))

        q_embeddings = self.model.encode(questions, normalize_embeddings=True, show_progress_bar=False)
        r_embeddings = self.model.encode(responses, normalize_embeddings=True, show_progress_bar=False)

        # Cosine similarity matrix (inner product of normalized vectors)
        alignment_matrix = np.dot(q_embeddings, r_embeddings.T)
        return alignment_matrix

    def verify_coverage(self, questions: List[str], candidate_responses: List[str]) -> Dict:
        """
        Executes out-of-order all-to-all semantic verification.
        Returns item-by-item audit details and overall compliance statistics.
        """
        if not questions:
            questions = ["Information sought by appellant"]

        if not candidate_responses:
            candidate_responses = ["No response statement available in narrative."]

        alignment_matrix = self.compute_alignment_matrix(questions, candidate_responses)
        num_q = len(questions)
        num_r = len(candidate_responses)

        item_records = []
        addressed_count = 0
        partial_count = 0
        missing_count = 0

        for i in range(num_q):
            q_text = questions[i]
            if num_r > 0:
                best_r_idx = int(np.argmax(alignment_matrix[i]))
                best_score = float(alignment_matrix[i, best_r_idx])
                best_r_text = candidate_responses[best_r_idx]
            else:
                best_r_idx = -1
                best_score = 0.0
                best_r_text = "None"

            # Threshold classification
            if best_score >= self.config.addressed_threshold:
                status = self.config.STATUS_ADDRESSED
                addressed_count += 1
            elif best_score >= self.config.partial_threshold:
                status = self.config.STATUS_PARTIAL
                partial_count += 1
            else:
                status = self.config.STATUS_MISSING
                missing_count += 1

            item_records.append({
                "item_id": f"Q{i+1}",
                "question_text": q_text,
                "best_match_sentence_idx": best_r_idx,
                "best_match_text": best_r_text,
                "similarity_score": round(best_score, 4),
                "status": status
            })

        # Calculate overall coverage ratio
        # Addressed = 1.0, Partial = 0.5, Missing = 0.0
        coverage_ratio = (addressed_count * 1.0 + partial_count * 0.5) / float(num_q)
        coverage_ratio = round(coverage_ratio, 4)

        # Determine overall case compliance verdict
        if addressed_count == num_q:
            compliance_verdict = self.config.VERDICT_FULL
        elif addressed_count > 0 or partial_count > 0:
            compliance_verdict = self.config.VERDICT_PARTIAL
        else:
            compliance_verdict = self.config.VERDICT_EVASION

        return {
            "total_items": num_q,
            "addressed_count": addressed_count,
            "partial_count": partial_count,
            "missing_count": missing_count,
            "coverage_ratio": coverage_ratio,
            "compliance_verdict": compliance_verdict,
            "embedding_model": self.model_name,
            "item_breakdown": item_records,
            "alignment_matrix": alignment_matrix.tolist()
        }
