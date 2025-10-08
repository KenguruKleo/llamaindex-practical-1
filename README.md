# LlamaIndex Candidate Explorer

This project ingests PDF resumes, chunks them into meaningful passages, embeds the passages with OpenAI embeddings, and writes them to a persistent ChromaDB vector store via LlamaIndex. Candidate metadata (name, profession, skills, summary) is produced by querying each Chroma-backed index with an OpenAI chat model. A Streamlit UI lists the indexed candidates and lets you drill into a dedicated profile page that surfaces summaries, key skills, and retrieved resume snippets.

## Requirements

- Python 3.12+
- A virtual environment (recommended)
- An OpenAI API key with access to the embedding and chat models you intend to use
- PDF resumes placed under `data/`

Create a virtual environment, activate it, and install dependencies (adjust the activation command for your shell/OS):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
# optional: echo "OPENAI_API_KEY=sk-..." > .env
```

Reactivate the environment in new shells with `source .venv/bin/activate` (or the corresponding Windows command) before running any project commands.

## Preparing the data

The first run of the web app will automatically process all PDFs in `data/`. To trigger indexing manually:

```bash
python -m app.data_pipeline
```

Generated artifacts are stored under `storage/`:

- `candidates.json` – metadata, summaries, and skills
- `chroma/` – persistent ChromaDB storage (single `candidates` collection containing every resume chunk)

If you add or update resumes, rebuild the index:

```bash
REBUILD_INDEX=1 python -m app.data_pipeline
```

## Running the web application

Serve the Streamlit UI:

```bash
streamlit run app/web.py
```

Streamlit prints the local URL in the terminal (typically http://localhost:8501). The landing page lists all indexed candidates with high-level metadata and summaries. Clicking `View full profile` opens a dedicated profile view that exposes the full summary, skill list, and generated highlights.

## Project layout

```
app/
  __init__.py        # prepare_candidates helper
  data_pipeline.py   # ingestion, chunking, embeddings, Chroma persistence
  web.py             # Streamlit entry point
static/
  styles.css         # legacy styling (unused by Streamlit UI; safe to remove)
```

## Notes

- Set `OPENAI_EMBED_MODEL` to switch to a different embedding model (defaults to `text-embedding-3-small`).
- Set `OPENAI_LLM_MODEL` to the chat model that extracts metadata and summaries (defaults to `gpt-4o-mini`).
- Make sure `OPENAI_API_KEY` is available in the environment (or `.env`) before running the pipeline or web app.
- ChromaDB stores vectors locally under `storage/chroma/`; remove this folder if you need a clean slate or to clear the shared `candidates` collection.
- Legacy Flask templates have been removed; the Streamlit app renders the UI directly in `app/web.py`.
