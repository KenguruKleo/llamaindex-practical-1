"""Candidate profiling application powered by LlamaIndex."""

from pathlib import Path
from typing import List

from .data_pipeline import CandidateIndexer, CandidateProfile, load_profiles

__all__ = [
    "CandidateIndexer",
    "CandidateProfile",
    "load_profiles",
]


def prepare_candidates(data_dir: Path, storage_dir: Path) -> List[CandidateProfile]:
    indexer = CandidateIndexer(data_dir=data_dir, storage_dir=storage_dir)
    return indexer.run()
