# OpenAI RAG

Set `OPENAI_API_KEY` in `.env`, then index `.txt` documents from `data/`:

```bash
uv run rag ingest
```

Ask questions against the saved Chroma index:

```bash
uv run rag ask "What do the documents say about the event store?"
```

Run `uv run rag ingest ./some-folder` to index a different folder. The default
Chroma database is `.rag/openai-chroma`; set `CHROMA_PATH` in `.env` to change
it. Ingest replaces the index. Ask reuses it and only embeds the question.
