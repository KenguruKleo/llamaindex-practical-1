# LlamaIndex Candidate Explorer

This project ingests PDF resumes, chunks them into meaningful passages, embeds the passages with OpenAI embeddings, and writes them to a persistent ChromaDB vector store via LlamaIndex. Candidate metadata (name, profession, skills, summary) is produced by querying each Chroma-backed index with an OpenAI chat model. A Flask UI lists the indexed candidates and exposes their summaries, key skills, and retrieved resume snippets.

## Requirements

- Python 3.12+
- A virtual environment (recommended)
- An OpenAI API key with access to the embedding and chat models you intend to use
- PDF resumes placed under `data/`

Install dependencies inside your virtualenv and configure your API key (for convenience you can place it in a `.env` file):

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
# optional: echo "OPENAI_API_KEY=sk-..." > .env
```

## Preparing the data

The first run of the web app will automatically process all PDFs in `data/`. To trigger indexing manually:

```bash
python -m app.data_pipeline
```

Generated artifacts are stored under `storage/`:

- `candidates.json` – metadata, summaries, and skills
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
  web.py             # Flask entry point
static/
  styles.css         # simple styling for the UI
templates/
  base.html, index.html, candidate.html
```

## Notes

- Set `OPENAI_EMBED_MODEL` to switch to a different embedding model (defaults to `text-embedding-3-small`).
- Set `OPENAI_LLM_MODEL` to the chat model that extracts metadata and summaries (defaults to `gpt-4o-mini`).
- Make sure `OPENAI_API_KEY` is available in the environment (or `.env`) before running the pipeline or web app.
- ChromaDB stores vectors locally under `storage/chroma/`; remove this folder if you need a clean slate.
