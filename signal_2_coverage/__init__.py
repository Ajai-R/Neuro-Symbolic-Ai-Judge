"""
Signal 2: Semantic Information Coverage Verification (FLAN-T5 + BGE-M3)
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

from .pipeline import CoverageVerificationPipeline
from .config import CoverageConfig

__all__ = ["CoverageVerificationPipeline", "CoverageConfig"]
