# Gemini RAG

This is the standalone Gemini version of the RAG CLI. It uses the project's
existing dependencies and `.env` file, while keeping its Chroma database
separate from other RAG implementations.

From the project root, add this to `.env` if it is not already set:

```dotenv
GOOGLE_GEMINI_API_KEY=your_key_here
```

Index documents and ask questions:

```bash
uv run python gemini/main.py ingest ./data
uv run python gemini/main.py ask "What do the documents say about?"
```

It supports `.txt`, `.md`, and `.pdf` files. The default database is
`gemini/.rag/chroma`. Override it with `--db`; choose another Gemini model with
`--model` on the `ask` command.
