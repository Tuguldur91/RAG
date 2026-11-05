import argparse
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .utils import CHROMA_PATH, DATA_PATH, load, load_into_vector_db, load_vector_db, split

load_dotenv()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}
---
Answer the question based on the above context: {question}
"""


def ask(question, db):
    matches = db.similarity_search(question, k=4)
    context = "\n\n---\n\n".join(document.page_content for document in matches)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    response = (prompt | model).invoke({"context": context, "question": question})

    print(response.content)
    print("\nSources:")
    for source in dict.fromkeys(
        document.metadata.get("source", "unknown") for document in matches
    ):
        print(f"- {source}")


def main():
    parser = argparse.ArgumentParser(
        prog="openai-rag", description="Document RAG using OpenAI."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="Rebuild the Chroma index")
    ingest_parser.add_argument("path", nargs="?", default=DATA_PATH)

    ask_parser = commands.add_parser("ask", help="Ask about indexed documents")
    ask_parser.add_argument("query", type=str, help="Question to ask")

    args = parser.parse_args()
    try:
        if args.command == "ingest":
            print("Loading and indexing documents…", flush=True)
            documents = load(args.path)
            load_into_vector_db(split(documents))
            print(f"Indexed documents in {CHROMA_PATH}")
        else:
            print("Loading saved Chroma index…", flush=True)
            db = load_vector_db()
            if db._collection.count() == 0:
                raise ValueError("The Chroma index is empty. Run `rag ingest` first.")
            print("Searching and generating answer…", flush=True)
            ask(args.query, db)
    except Exception as error:
        parser.exit(1, f"rag: error: {error}\n")


if __name__ == "__main__":
    main()
