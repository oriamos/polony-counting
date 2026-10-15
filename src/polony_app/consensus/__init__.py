"""Consensus between detection engines (spatial matching only; agreement is not ground truth)."""
from .matching import ConsensusPair, match_engines, summarize

__all__ = ["ConsensusPair", "match_engines", "summarize"]
