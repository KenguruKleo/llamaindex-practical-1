from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import chromadb

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.vector_stores.chroma import ChromaVectorStore

from .data_pipeline import (
    CHROMA_COLLECTION,
    CandidateProfile,
    create_embedding_model,
    create_llm,
    load_profiles,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"


@lru_cache(maxsize=1)
def _load_profiles() -> List[CandidateProfile]:
    return load_profiles(STORAGE_DIR)


@lru_cache(maxsize=1)
def _load_index() -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=create_embedding_model(),
    )


def _build_retrieval_tool(index: VectorStoreIndex) -> QueryEngineTool:
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="candidate_retriever",
            description=(
                "Use this tool to answer questions about candidates, their experience "
                "and professional highlights extracted from resumes."
                "Include profile ID, name, profession, skills, and summary where appropriate."
                "Format it as a concise bullet list.!"
            ),
        ),
    )


def _format_profile(profile: CandidateProfile) -> str:
    parts = [f"Name: {profile.name}", f"Profession: {profile.profession}"]
    if profile.years_experience:
        parts.append(f"Experience: {profile.years_experience} years")
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.summary:
        parts.append(f"Summary: {profile.summary}")
    parts.append(f"Source: {profile.source_file}")
    return "\n".join(parts)


def _profile_lookup_tool(profiles: Iterable[CandidateProfile]) -> FunctionTool:
    name_to_profile = {profile.name.lower(): profile for profile in profiles}
    id_to_profile = {profile.id: profile for profile in profiles}

    def _lookup(identifier: str) -> str:
        if not identifier:
            return "Please specify a candidate name or identifier."
        key = identifier.strip().lower()
        profile = name_to_profile.get(key) or id_to_profile.get(key)
        if not profile:
            return (
                "Candidate not found. Try using the candidate's exact name or their ID "
                "from the directory."
            )
        return _format_profile(profile)

    return FunctionTool.from_defaults(
        fn=_lookup,
        name="profile_lookup",
        description=(
            "Look up a candidate by name or identifier to retrieve their profile "
            "information such as profession, skills, and summary."
        ),
    )


@lru_cache(maxsize=1)
def get_agent() -> ReActAgent:
    index = _load_index()
    profiles = _load_profiles()
    tools = [
        _build_retrieval_tool(index),
        _profile_lookup_tool(profiles),
        # _skill_search_tool(profiles),
    ]
    llm = create_llm()
    return ReActAgent(tools=tools, llm=llm, verbose=False, streaming=False)


def _run_agent_sync(
    agent: ReActAgent,
    message: str,
    chat_history: Optional[List[ChatMessage]] = None,
) -> AgentOutput:
    async def _arun() -> AgentOutput:
        handler = agent.run(user_msg=message, chat_history=chat_history)
        return await handler

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_arun())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
    else:  # pragma: no cover - only triggered if a loop already runs
        future = asyncio.run_coroutine_threadsafe(_arun(), loop)
        return future.result()


def chat_with_agent(
    message: str,
    chat_history: Optional[List[ChatMessage]] = None,
) -> AgentOutput:
    agent = get_agent()
    return _run_agent_sync(agent, message, chat_history)


__all__ = ["chat_with_agent", "get_agent"]
