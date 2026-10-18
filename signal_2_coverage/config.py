"""
Configuration and Thresholds for Signal 2: Semantic Information Coverage Verification
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import os
from dataclasses import dataclass


@dataclass
class CoverageConfig:
    # Dense Semantic Embedding Models
    # Primary: BAAI/bge-m3 (8192 context length, state-of-the-art dense embeddings)
    # Fallback: all-MiniLM-L6-v2 (locally cached, lightweight, rapid inference)
    primary_embedding_model: str = "BAAI/bge-m3"
    fallback_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Evidence Extraction Models
    primary_extraction_model: str = "google/flan-t5-base"
    fallback_extraction_model: str = "google/flan-t5-small"
    
    # Calibrated Semantic Similarity Thresholds
    # S >= 0.70: The PIO directly provided the requested information (ADDRESSED)
    # 0.45 <= S < 0.70: The PIO gave an ambiguous, partial, or evasive response (PARTIALLY_ADDRESSED)
    # S < 0.45: The PIO completely ignored or omitted this question item (MISSING)
    addressed_threshold: float = 0.70
    partial_threshold: float = 0.45
    
    # Compliance Category Labels
    STATUS_ADDRESSED: str = "ADDRESSED"
    STATUS_PARTIAL: str = "PARTIALLY_ADDRESSED"
    STATUS_MISSING: str = "MISSING"
    
    # Overall Case Verdicts
    VERDICT_FULL: str = "FULL_COMPLIANCE"
    VERDICT_PARTIAL: str = "PARTIAL_COMPLIANCE"
    VERDICT_EVASION: str = "TOTAL_EVASION"
    
    # Computation settings
    max_seq_length: int = 512
    device: str = "cuda"  # Auto-detects cuda if available, else cpu
