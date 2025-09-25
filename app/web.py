from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

import chromadb
from flask import Flask, abort, render_template
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from . import CandidateProfile, load_profiles, prepare_candidates
from .data_pipeline import create_embedding_model

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)

if os.getenv("REBUILD_INDEX", "0") == "1" or not (STORAGE_DIR / "candidates.json").exists():
    prepare_candidates(DATA_DIR, STORAGE_DIR)

PROFILES: List[CandidateProfile] = load_profiles(STORAGE_DIR)
PROFILE_LOOKUP = {profile.id: profile for profile in PROFILES}

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


@lru_cache(maxsize=32)
def load_index(candidate_id: str) -> VectorStoreIndex:
    collection = chroma_client.get_or_create_collection(candidate_id)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    candidate_dir = STORAGE_DIR / candidate_id
    if not candidate_dir.exists():
        raise FileNotFoundError(f"No persisted index found for candidate '{candidate_id}'")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(candidate_dir),
    )
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=create_embedding_model(),
    )


@app.route("/")
def index() -> str:
    return render_template("index.html", profiles=PROFILES)


@app.route("/candidate/<candidate_id>")
def candidate_view(candidate_id: str) -> str:
    profile = PROFILE_LOOKUP.get(candidate_id)
    if not profile:
        abort(404)

    snippets = []
    try:
        index = load_index(candidate_id)
        retriever = index.as_retriever(similarity_top_k=3)
        results = retriever.retrieve("professional highlights and core skills")
        seen = set()
        for item in results:
            text = item.node.text.strip() # type: ignore
            if text and text not in seen:
                seen.add(text)
                snippets.append(text)
    except Exception:
        snippets = []

    return render_template(
        "candidate.html",
        profile=profile,
        snippets=snippets,
    )


@app.context_processor
def inject_globals():
    return {"total_candidates": len(PROFILES)}


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    app.run(debug=True, port=8000)
