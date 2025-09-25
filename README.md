# LlamaIndex Candidate Explorer

This project ingests PDF resumes, chunks them into meaningful passages, embeds the passages with an Ollama-hosted model, and writes them to a persistent ChromaDB vector store via LlamaIndex. A Flask UI lists the indexed candidates and exposes their summaries, key skills, and retrieved resume snippets.

## Requirements

- Python 3.12+
- A virtual environment (recommended)
- Ollama running locally with an embedding model available (e.g. `nomic-embed-text`)
- PDF resumes placed under `data/`

Install dependencies (inside your virtualenv) and make sure the embed model is pulled:

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
```

## Preparing the data

The first run of the web app will automatically process all PDFs in `data/`. To trigger indexing manually:

```bash
python -m app.data_pipeline
```

Generated artifacts are stored under `storage/`:

- `candidates.json` – metadata, summaries, and skill highlights
- `chroma/` – persistent ChromaDB collection per candidate
- `<candidate-id>/` – LlamaIndex storage context for each resume

If you add or update resumes, rebuild the index:

```bash
REBUILD_INDEX=1 python -m app.web
```

## Running the web application

Ensure the Ollama service is running (`ollama serve`) and start the development server with Flask's CLI:

```bash
FLASK_APP=app.web:create_app flask run --reload
```

The UI will be available at http://127.0.0.1:5000/ (or the host/port you configure). The home page lists all indexed candidates; selecting one reveals their extracted details, generated summary, skill pills, and retrieved passages sourced from the vector store.

## Project layout

```
app/
  __init__.py        # prepare_candidates helper
  data_pipeline.py   # ingestion, chunking, embeddings, Chroma persistence
  web.py             # Flask entry point
static/
  styles.css         # simple styling for the UI
templates/
  base.html, index.html, candidate.html
```

## Notes

- Set `OLLAMA_EMBED_MODEL` to switch to a different local embedding model (defaults to `nomic-embed-text`).
- The skill extraction heuristics are keyword-based; adjust `SKILL_KEYWORDS` in `app/data_pipeline.py` to fit your domain.
- ChromaDB stores vectors locally under `storage/chroma/`; remove this folder if you need a clean slate.
