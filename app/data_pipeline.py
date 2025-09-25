from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import chromadb

try:
    from llama_index.readers.file import SimpleDirectoryReader
except ImportError:  # pragma: no cover - compatibility with older packages
    from llama_index.core import SimpleDirectoryReader  # type: ignore

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore

from .embeddings import HashEmbedding


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


SKILL_KEYWORDS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "sql",
    "postgresql",
    "mongodb",
    "react",
    "django",
    "flask",
    "fastapi",
    "machine learning",
    "deep learning",
    "nlp",
    "llm",
}


class CandidateIndexer:
    def __init__(self, data_dir: Path, storage_dir: Path) -> None:
        self._data_dir = data_dir
        self._storage_dir = storage_dir
        self._chroma_dir = self._storage_dir / "chroma"
        self._embed_model = HashEmbedding()
        self._splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)
        self._chroma_client = chromadb.PersistentClient(path=str(self._chroma_dir))

    def run(self) -> List[CandidateProfile]:
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        profiles: List[CandidateProfile] = []

        for pdf_path in sorted(self._data_dir.glob("*.pdf")):
            candidate_id = slugify(pdf_path.stem)
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
        name = extract_name(full_text)
        profession = extract_profession(full_text)
        years = extract_years_of_experience(full_text)
        skills = extract_skills(full_text)
        summary, highlights = build_summary(index)

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

    def _persist_profiles(self, profiles: List[CandidateProfile]) -> None:
        payload = [profile.to_json() for profile in profiles]
        output_path = self._storage_dir / "candidates.json"
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)


def slugify(value: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9]+", value.lower())
    return "-".join(tokens) or "candidate"


def extract_name(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    # Many CVs open with the person's name in the first few lines
    for line in lines[:5]:
        if len(line.split()) <= 5 and line.replace(" ", "").isalpha():
            return line.title()
    return lines[0][:80]


def extract_profession(text: str) -> Optional[str]:
    pattern = re.compile(r"(senior|lead|principal|full\s*stack|software|data|machine learning|devops)[^\n]{0,60}", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(0).strip().replace("\n", " ")
    return None


def extract_years_of_experience(text: str) -> Optional[float]:
    matches = re.findall(r"(\d{1,2})\+?\s+(?:years|yrs)", text, flags=re.IGNORECASE)
    if matches:
        values = [float(m) for m in matches]
        return max(values)
    return None


def extract_skills(text: str, top_k: int = 10) -> List[str]:
    text_lower = text.lower()
    counts: Dict[str, int] = {}
    for keyword in SKILL_KEYWORDS:
        occurrences = text_lower.count(keyword)
        if occurrences:
            counts[keyword] = occurrences
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [skill for skill, _ in ranked[:top_k]]


def build_summary(index: VectorStoreIndex, max_sentences: int = 5) -> tuple[str, List[str]]:
    retriever = index.as_retriever(similarity_top_k=6)
    results = retriever.retrieve("strongest skills and professional highlights")

    sentences: List[str] = []
    for item in results:
        chunk_sentences = split_sentences(item.node.text)
        for sentence in chunk_sentences:
            cleaned = sentence.strip()
            if cleaned and cleaned not in sentences:
                sentences.append(cleaned)

    summary = compose_summary(sentences, max_sentences=max_sentences)
    highlights = sentences[: min(3, len(sentences))]
    return summary, highlights


def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def compose_summary(sentences: Iterable[str], max_sentences: int = 5) -> str:
    sentences = list(sentences)
    if not sentences:
        return "Summary not available."

    word_counts = Counter()
    for sentence in sentences:
        tokens = re.findall(r"\b\w+\b", sentence.lower())
        word_counts.update(tokens)

    scored = []
    for sentence in sentences:
        tokens = re.findall(r"\b\w+\b", sentence.lower())
        score = sum(word_counts[token] for token in tokens)
        scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_sentences = [sentence for _, sentence in scored[:max_sentences]]
    return " ".join(top_sentences)


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
