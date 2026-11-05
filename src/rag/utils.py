import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data")
CHROMA_PATH = os.getenv("CHROMA_PATH", ".rag/openai-chroma")


def load(path="data"):
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Data path does not exist: {source}")

    if source.is_file():
        documents = TextLoader(str(source), encoding="utf-8").load()
        for document in documents:
            document.metadata["source"] = str(source.resolve())
        return documents

    loader = DirectoryLoader(
        path=str(source),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"No .txt files found in {source}")
    return documents


def split(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)


def load_into_vector_db(chunks):
    if not chunks:
        raise ValueError("No document chunks to index.")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="openai_documents",
    )


def load_vector_db():
    if not Path(CHROMA_PATH).exists():
        raise FileNotFoundError(
            f"Chroma database not found at {CHROMA_PATH}. Run `rag ingest` first."
        )
    return Chroma(
        collection_name="openai_documents",
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings(),
    )
