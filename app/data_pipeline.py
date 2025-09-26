from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import chromadb
from dotenv import load_dotenv

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

try:
    from llama_index.core.readers import SimpleDirectoryReader
except ImportError:  # pragma: no cover - compatibility with older packages
    try:
        from llama_index.readers.file import SimpleDirectoryReader  # type: ignore
    except ImportError:
        from llama_index.core import SimpleDirectoryReader  # type: ignore

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

CHROMA_COLLECTION = "candidates"


def create_embedding_model() -> BaseEmbedding:
    return OpenAIEmbedding(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)


def create_llm() -> LLM:
    return OpenAI(model=OPENAI_LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)


def extract_candidate_details(
    index: VectorStoreIndex,
    *,
    llm: LLM,
    top_k: int = 8,
    filters: Optional[MetadataFilters] = None,
) -> Dict[str, object]:
    prompt = (
        "You are analysing a software engineer resume. Extract the candidate's name, "
        "profession, a concise list of skills, total years of experience, and a summary "
        "of strongest skills, achievements, and professional highlights.\n\n"
        "Respond strictly as JSON with this schema:\n"
        "{\n"
        "  \"name\": \"...\",\n"
        "  \"profession\": \"...\",\n"
        "  \"skills\": [\"...\", \"...\"],\n"
        "  \"years_experience\": \"...\",\n"
        "  \"summary\": \"...\"\n"
        "}"
    )
    try:
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            llm=llm,
            response_mode="compact",
            filters=filters,
        )
        response = query_engine.query(prompt)
    except Exception as ex:
        print(f"Error extracting candidate details: {ex}")
        return {}

    raw = str(getattr(response, "response", response))
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}

def _sanitize_skills(raw: object, *, limit: int = 10) -> List[str]:
    if isinstance(raw, list):
        cleaned = [str(item).strip() for item in raw if isinstance(item, (str, int, float))]
        return [skill for skill in cleaned if skill][:limit]
    if isinstance(raw, str):
        candidates = [part.strip() for part in raw.split(",") if part.strip()]
        return candidates[:limit]
    return []


@dataclass
class CandidateProfile:
    id: str
    name: str
    profession: str
    years_experience: Optional[str]
    summary: str
    skills: List[str]
    source_file: str

    def to_json(self) -> Dict[str, object]:
        payload = asdict(self)
        return payload

class CandidateIndexer:
    def __init__(
        self,
        data_dir: Path,
        storage_dir: Path,
    ) -> None:
        self._data_dir = data_dir
        self._storage_dir = storage_dir
        self._chroma_dir = self._storage_dir / "chroma"
        self._embed_model = create_embedding_model()
        self._llm = create_llm()
        self._splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)
        self._chroma_client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._collection_name = CHROMA_COLLECTION

    def run(self) -> List[CandidateProfile]:
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        profiles: List[CandidateProfile] = []

        for pdf_path in sorted(self._data_dir.glob("*.pdf")):
            candidate_id = slugify(pdf_path.stem)
            self._reset_candidate_storage(candidate_id)
            documents = self._load_documents(pdf_path, candidate_id)
            nodes = self._build_nodes(documents, candidate_id, pdf_path.name)

            profile = self._build_index(candidate_id, nodes, documents, pdf_path.name)
            profiles.append(profile)

        self._persist_profiles(profiles)
        return profiles

    def _load_documents(self, path: Path, candidate_id: str) -> List[Document]:
        reader = SimpleDirectoryReader(input_files=[str(path)], filename_as_id=True)
        documents = reader.load_data()
        for doc in documents:
            doc.metadata.setdefault("candidate_id", candidate_id)
            doc.metadata.setdefault("source_file", path.name)
        return documents

    def _build_nodes(self, documents: Iterable[Document], candidate_id: str, source_file: str):
        nodes = self._splitter.get_nodes_from_documents(list(documents))
        for node in nodes:
            node.metadata.setdefault("candidate_id", candidate_id)
            node.metadata.setdefault("source_file", source_file)
        return nodes

    def _build_index(
        self,
        candidate_id: str,
        nodes,
        _documents: List[Document],
        source_file: str,
    ) -> CandidateProfile:
        collection = self._chroma_client.get_or_create_collection(name=self._collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self._embed_model)

        filters = MetadataFilters(filters=[MetadataFilter(key="candidate_id", value=candidate_id)])
        details = extract_candidate_details(index, llm=self._llm, filters=filters)

        raw_skills = details.get("skills") or details.get("skils")
        skills = _sanitize_skills(raw_skills)
        name = str(details.get("name", "")).strip()
        profession = str(details.get("profession", "")).strip()
        years_experience = str(details.get("years_experience"))
        summary = str(details.get("summary", "")).strip()

        return CandidateProfile(
            id=candidate_id,
            name=name or candidate_id.replace("-", " ").title(),
            profession=profession or "Unknown",
            years_experience=years_experience,
            summary=summary,
            skills=skills,
            source_file=source_file,
        )

    def _reset_candidate_storage(self, candidate_id: str) -> None:
        candidate_dir = self._storage_dir / candidate_id
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir, ignore_errors=True)
        try:
            collection = self._chroma_client.get_or_create_collection(name=self._collection_name)
            collection.delete(where={"candidate_id": candidate_id})
        except Exception:
            pass

    def _persist_profiles(self, profiles: List[CandidateProfile]) -> None:
        payload = [profile.to_json() for profile in profiles]
        output_path = self._storage_dir / "candidates.json"
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)


def slugify(value: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9]+", value.lower())
    return "-".join(tokens) or "candidate"

def load_profiles(storage_dir: Path) -> List[CandidateProfile]:
    data_path = storage_dir / "candidates.json"
    if not data_path.exists():
        return []
    with data_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    profiles: List[CandidateProfile] = []
    for item in raw:
        profiles.append(
            CandidateProfile(
                id=item["id"],
                name=item["name"],
                profession=item.get("profession", "Unknown"),
                years_experience=item.get("years_experience"),
                summary=item.get("summary", ""),
                skills=item.get("skills", []),
                source_file=item.get("source_file", ""),
            )
        )
    return profiles


if __name__ == "__main__":  # pragma: no cover - convenience script
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    storage_dir = base_dir / "storage"
    indexer = CandidateIndexer(data_dir=data_dir, storage_dir=storage_dir)
    profiles = indexer.run()
    print(f"Indexed {len(profiles)} candidate(s) into {storage_dir}.")
