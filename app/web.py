from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent

import chromadb
import streamlit as st
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

try:
    from . import CandidateProfile, load_profiles, prepare_candidates
    from .data_pipeline import create_embedding_model
except ImportError:  # when executed as a script via ``streamlit run``
    import sys

    sys.path.append(str(BASE_DIR))
    from app import CandidateProfile, load_profiles, prepare_candidates
    from app.data_pipeline import create_embedding_model

DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)

if os.getenv("REBUILD_INDEX", "0") == "1" or not (STORAGE_DIR / "candidates.json").exists():
    prepare_candidates(DATA_DIR, STORAGE_DIR)

PROFILES: List[CandidateProfile] = load_profiles(STORAGE_DIR)
PROFILE_LOOKUP = {profile.id: profile for profile in PROFILES}

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))


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


@st.cache_data(show_spinner=False)
def fetch_candidate_snippets(candidate_id: str) -> List[str]:
    try:
        index = load_index(candidate_id)
        retriever = index.as_retriever(similarity_top_k=3)
        results = retriever.retrieve("professional highlights and core skills")
    except Exception:
        return []

    snippets: List[str] = []
    seen = set()
    for item in results:
        text = str(getattr(item.node, "text", "")).strip()
        if text and text not in seen:
            seen.add(text)
            snippets.append(text)
    return snippets


def _format_summary(summary: str, limit: int = 240) -> str:
    summary = summary.strip()
    if len(summary) <= limit:
        return summary
    return summary[:limit].rstrip() + "…"


def _set_candidate_page(candidate_id: Optional[str]) -> None:
    if candidate_id:
        st.query_params["candidate"] = candidate_id
    else:
        st.query_params.pop("candidate", None)
    st.rerun()


def render_directory(profiles: List[CandidateProfile], active_candidate_id: Optional[str]) -> None:
    st.header("Candidate Directory")
    if not profiles:
        st.info("No candidates processed. Add PDF resumes to the data directory and rebuild the index.")
        return

    for profile in profiles:
        with st.container(border=True):
            st.markdown(f"### {profile.name}")
            st.caption(profile.profession)
            if profile.years_experience:
                st.write(f"Experience: {profile.years_experience} years")
            if profile.summary:
                st.write(_format_summary(profile.summary))

            button_disabled = profile.id == active_candidate_id
            button_label = "Viewing profile" if button_disabled else "View full profile"
            if st.button(
                button_label,
                key=f"select-{profile.id}",
                use_container_width=True,
                disabled=button_disabled,
            ):
                _set_candidate_page(profile.id)


def render_candidate_details(profile: CandidateProfile) -> None:
    if st.button("← Back to directory", use_container_width=False):
        _set_candidate_page(None)

    st.markdown(f"### {profile.name}")
    st.caption(profile.profession)

    meta_items = []
    if profile.years_experience:
        meta_items.append(f"Experience: {profile.years_experience} years")
    meta_items.append(f"Source: {profile.source_file}")
    st.write(" | ".join(meta_items))

    if profile.summary:
        st.markdown("#### Summary")
        st.write(profile.summary)

    if profile.skills:
        st.markdown("#### Skills")
        st.write(", ".join(profile.skills))

    st.markdown("#### Highlights")
    with st.spinner("Retrieving top snippets…"):
        snippets = fetch_candidate_snippets(profile.id)

    if snippets:
        for snippet in snippets:
            st.markdown(f"- {snippet}")
    else:
        st.info("No highlights available. The index might still be building or the resume lacks detailed sections.")


def main() -> None:
    st.set_page_config(page_title="Candidate Explorer", page_icon="🧑‍💼", layout="wide")

    st.sidebar.title("Candidate Explorer")
    st.sidebar.metric("Total candidates", len(PROFILES))

    params = st.query_params
    candidate_value = params.get("candidate")
    if isinstance(candidate_value, list):
        current_id = candidate_value[0]
    elif isinstance(candidate_value, str):
        current_id = candidate_value
    else:
        current_id = None

    selected_profile = PROFILE_LOOKUP.get(current_id) if current_id else None

    if selected_profile:
        st.sidebar.success(f"Viewing: {selected_profile.name}")
        st.divider()
        render_candidate_details(selected_profile)
    else:
        render_directory(PROFILES, current_id)


if __name__ == "__main__":
    main()
