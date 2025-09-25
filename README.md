# LlamaIndex Candidate Explorer

This project ingests PDF resumes, chunks them into meaningful passages, embeds the passages with a lightweight deterministic model, and writes them to a persistent ChromaDB vector store via LlamaIndex. A Flask UI lists the indexed candidates and exposes their summaries, key skills, and retrieved resume snippets.

## Requirements

- Python 3.12+
- A virtual environment (recommended)
- PDF resumes placed under `data/`

Install dependencies:

```bash
pip install -r requirements.txt
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

Start the development server with Flask's CLI:

```bash
FLASK_APP=app.web:create_app flask run --reload
```

The UI will be available at http://127.0.0.1:5000/ (or the host/port you configure). The home page lists all indexed candidates; selecting one reveals their extracted details, generated summary, skill pills, and retrieved passages sourced from the vector store.

## Project layout

```
app/
  __init__.py        # prepare_candidates helper
  data_pipeline.py   # ingestion, chunking, embeddings, Chroma persistence
  embeddings.py      # hash-based embedding implementation
  web.py             # Flask entry point
static/
  styles.css         # simple styling for the UI
templates/
  base.html, index.html, candidate.html
```

## Notes

- `HashEmbedding` delivers deterministic embeddings without external APIs or network access, keeping the pipeline self-contained.
- The skill extraction heuristics are keyword-based; adjust `SKILL_KEYWORDS` in `app/data_pipeline.py` to fit your domain.
- ChromaDB stores vectors locally under `storage/chroma/`; remove this folder if you need a clean slate.
