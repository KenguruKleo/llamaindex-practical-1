from __future__ import annotations

import os
import base64
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent

import chromadb
import streamlit as st
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import ChatMessage

try:
    from . import CHROMA_COLLECTION, CandidateProfile, load_profiles, prepare_candidates
    from .agent import chat_with_agent
    from .data_pipeline import create_embedding_model
except ImportError:  # when executed as a script via ``streamlit run``
    import sys

    sys.path.append(str(BASE_DIR))
    from app import CHROMA_COLLECTION, CandidateProfile, load_profiles, prepare_candidates
    from app.agent import chat_with_agent
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


@lru_cache(maxsize=1)
def load_index() -> VectorStoreIndex:
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=create_embedding_model(),
    )

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
            if profile.skills:
                st.write(f"Skills: {', '.join(profile.skills)}")
            if profile.summary:
                st.write(_format_summary(profile.summary, 1000))

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

    pdf_path = DATA_DIR / f"{profile.id}.pdf"
    st.markdown("#### Resume")
    if pdf_path.exists():
        with pdf_path.open("rb") as fp:
            pdf_bytes = fp.read()
        encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        st.download_button(
            label="Download resume PDF",
            data=pdf_bytes,
            file_name=pdf_path.name,
            mime="application/pdf",
        )
        st.markdown(
            f"""
            <div style="border:1px solid #ddd; height:600px;">
              <iframe
                src="data:application/pdf;base64,{encoded_pdf}#toolbar=1"
                style="width:100%; height:100%; border:0;"
                title="Resume preview"
              ></iframe>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Resume PDF not found in the data directory.")


def render_agent_chat() -> None:
    st.markdown("### Candidate Assistant Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    prompt = st.chat_input("Ask the assistant about candidates or skills...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            history: List[ChatMessage] = st.session_state.messages
            agent_output = chat_with_agent(prompt, chat_history=history.copy())
            response = agent_output.response.content or ""
            # enrich display with tool usage when available
            if agent_output.tool_calls:
                tools_used = ", ".join(
                    call.tool_name for call in agent_output.tool_calls
                )
                response += f"\n\n_Tools used: {tools_used}_"
            else:
                response += "\n\n_No embedded tools used._"

            st.session_state.messages.append({"role": "assistant", "content": response})
        # rerender the chat to show the new messages
        st.rerun()


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

    tab_directory, tab_agent = st.tabs(["Candidate Directory", "Agent Chat"])

    with tab_directory:
        if selected_profile:
            render_candidate_details(selected_profile)
        else:
            render_directory(PROFILES, current_id)

    with tab_agent:
        render_agent_chat()


if __name__ == "__main__":
    main()
