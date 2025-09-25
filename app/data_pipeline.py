from __future__ import annotations

import json
import os
import re
import shutil
from functools import lru_cache
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import chromadb
import requests

try:
    from llama_index.core.readers import SimpleDirectoryReader
except ImportError:  # pragma: no cover - compatibility with older packages
    try:
        from llama_index.readers.file import SimpleDirectoryReader  # type: ignore
    except ImportError:
        from llama_index.core import SimpleDirectoryReader  # type: ignore

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434").rstrip("/")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")


@lru_cache(maxsize=4)
def _cached_embedding_model(model_name: str) -> BaseEmbedding:
    return OllamaEmbedding(model_name=model_name)


def create_embedding_model(model_name: str | None = None) -> BaseEmbedding:
    target_model = model_name or DEFAULT_OLLAMA_MODEL
    return _cached_embedding_model(target_model)


def _ollama_generate(prompt: str, *, model: str = OLLAMA_LLM_MODEL, timeout: int = 45) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def _extract_json_object(raw: str) -> Dict[str, object]:
    snippet = raw.strip()
    if "```" in snippet:
        parts = snippet.split("```")
        candidates = [part for part in parts if "{" in part and "}" in part]
        if candidates:
            snippet = max(candidates, key=len)
    start = snippet.find("{")
    end = snippet.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(snippet[start : end + 1])
    except json.JSONDecodeError:
        return {}


def extract_candidate_details(text: str, *, max_chars: int = 6000) -> Dict[str, object]:
    excerpt = text[:max_chars]
    prompt = (
        "Extract from this resume only:\n"
        "- Name\n"
        "- Skills\n"
        "- Profession\n"
        "- Years of experience\n"
        "And summarize strongest skills, achievements, and professional highlights.\n\n"
        "Extract candidate details in JSON strictly following this schema:\n"
        "{\n"
        "  \"name\": \"...\",\n"
        "  \"profession\": \"...\",\n"
        "  \"skills\": [\"...\", \"...\"],\n"
        "  \"years_experience\": \"...\",\n"
        "  \"summary\": \"...\"\n"
        "}\n\n"
        f"Resume text:\n{excerpt}"
    )
    try:
        raw = _ollama_generate(prompt)
        return _extract_json_object(raw)
    except Exception:
        return {}


def _parse_years_experience(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


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
    years_experience: Optional[float]
    summary: str
    skills: List[str]
    highlights: List[str]
    source_file: str

    def to_json(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["years_experience"] = (
            round(self.years_experience, 1) if self.years_experience is not None else None
        )
        return payload

class CandidateIndexer:
    def __init__(
        self,
        data_dir: Path,
        storage_dir: Path,
        *,
        embed_model: BaseEmbedding | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._storage_dir = storage_dir
        self._chroma_dir = self._storage_dir / "chroma"
        self._embed_model = embed_model or create_embedding_model()
        self._splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)
        self._chroma_client = chromadb.PersistentClient(path=str(self._chroma_dir))

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
        documents: List[Document],
        source_file: str,
    ) -> CandidateProfile:
        collection = self._chroma_client.get_or_create_collection(name=candidate_id)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self._embed_model)
        candidate_dir = self._storage_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(candidate_dir))

        full_text = "\n".join(doc.text for doc in documents)
        details = extract_candidate_details(full_text)

        raw_skills = details.get("skills") or details.get("skils")
        skills = _sanitize_skills(raw_skills)
        name = str(details.get("name", "")).strip()
        profession = str(details.get("profession", "")).strip()
        years = _parse_years_experience(details.get("years_experience"))
        summary = str(details.get("summary", "")).strip()
        highlights = retrieve_highlights(index)

        return CandidateProfile(
            id=candidate_id,
            name=name or candidate_id.replace("-", " ").title(),
            profession=profession or "Unknown",
            years_experience=years,
            summary=summary,
            skills=skills,
            highlights=highlights,
            source_file=source_file,
        )

    def _reset_candidate_storage(self, candidate_id: str) -> None:
        candidate_dir = self._storage_dir / candidate_id
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir, ignore_errors=True)
        try:
            self._chroma_client.delete_collection(name=candidate_id)
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


def retrieve_highlights(index: VectorStoreIndex, max_sentences: int = 3) -> List[str]:
    retriever = index.as_retriever(similarity_top_k=6)
    results = retriever.retrieve("strongest skills and professional highlights")

    sentences: List[str] = []
    seen = set()
    for item in results:
        chunk_sentences = split_sentences(item.node.text) # type: ignore
        for sentence in chunk_sentences:
            cleaned = sentence.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                sentences.append(cleaned)
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= max_sentences:
            break
    return sentences


def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)

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
                highlights=item.get("highlights", []),
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
