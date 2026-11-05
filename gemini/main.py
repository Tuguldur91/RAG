"""Gemini version of the local document RAG CLI.

Run from the project root with:
    uv run python gemini/main.py ingest ./data
    uv run python gemini/main.py ask "What is in the documents?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = str(Path(__file__).resolve().parent / ".rag" / "chroma")
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def api_key() -> str:
    """Load a Gemini key without ever logging its value."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = (
        os.getenv("GOOGLE_GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "Gemini API key not found. Add GOOGLE_GEMINI_API_KEY to the project .env file."
        )
    return key


def vector_store(db_dir: str) -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=api_key()
    )
    return Chroma(
        collection_name="gemini_documents",
        persist_directory=db_dir,
        embedding_function=embeddings,
    )


def load_file(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        documents = PyPDFLoader(str(path)).load()
    else:
        documents = TextLoader(
            str(path), encoding="utf-8", autodetect_encoding=True
        ).load()
    for document in documents:
        document.metadata["source"] = str(path.resolve())
    return documents


def ingest(source: str, db_dir: str, chunk_size: int, chunk_overlap: int) -> int:
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source does not exist: {source_path}")

    if source_path.is_file():
        files = [source_path] if source_path.suffix.lower() in SUPPORTED_SUFFIXES else []
    else:
        files = sorted(
            path for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    if not files:
        raise ValueError("No .txt, .md, or .pdf files found in the source.")

    documents: list[Document] = []
    for path in files:
        try:
            documents.extend(load_file(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
    if not documents:
        raise ValueError("No readable documents found.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Embedding and saving {len(chunks)} chunks…", flush=True)
    vector_store(db_dir).add_documents(chunks)
    return len(chunks)


def ask(question: str, db_dir: str, k: int, model: str) -> None:
    print("Loading knowledge base…", flush=True)
    store = vector_store(db_dir)
    if store._collection.count() == 0:
        raise ValueError("The knowledge base is empty. Run the ingest command first.")

    print("Searching documents (Gemini query embedding)…", flush=True)
    matches = store.similarity_search(question, k=k)
    if not matches:
        raise ValueError("No matching document chunks were found.")

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')} | "
        f"page: {doc.metadata.get('page', 'n/a')}]\n{doc.page_content}"
        for doc in matches
    )
    print(f"Generating answer with {model}…", flush=True)
    llm = ChatGoogleGenerativeAI(
        model=model, google_api_key=api_key(), temperature=0
    )
    response = llm.invoke(
        "Answer using only the context below. If the answer is not present, say "
        "you don't know based on these documents. Cite source names and pages "
        "when available.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    print(response.content)
    print("\nSources:")
    shown: set[tuple[str, object]] = set()
    for doc in matches:
        source = str(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")
        citation = (source, page)
        if citation not in shown:
            shown.add(citation)
            print(f"- {source}" + (f", page {page}" if page is not None else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gemini-rag", description="Local document RAG using Gemini and Chroma."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="Index .txt, .md, and .pdf files")
    ingest_parser.add_argument("source", help="A file or directory to index")
    ingest_parser.add_argument("--db", default=DEFAULT_DB)
    ingest_parser.add_argument("--chunk-size", type=int, default=1000)
    ingest_parser.add_argument("--chunk-overlap", type=int, default=150)

    ask_parser = commands.add_parser("ask", help="Ask a question about indexed files")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--db", default=DEFAULT_DB)
    ask_parser.add_argument("-k", type=int, default=4)
    ask_parser.add_argument(
        "--model", default=os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    )

    args = parser.parse_args()
    try:
        if args.command == "ingest":
            if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
                parser.error("chunk overlap must be nonnegative and smaller than chunk size")
            count = ingest(args.source, args.db, args.chunk_size, args.chunk_overlap)
            print(f"Indexed {count} chunks in {args.db}")
        else:
            if args.k <= 0:
                parser.error("-k must be greater than zero")
            ask(args.question, args.db, args.k, args.model)
    except Exception as exc:
        parser.exit(1, f"gemini-rag: error: {exc}\n")


if __name__ == "__main__":
    main()
