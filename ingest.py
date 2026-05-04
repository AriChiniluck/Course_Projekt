"""
Knowledge ingestion pipeline.

Loads documents from data/ directory, splits into chunks,
generates embeddings, and saves the index to disk.

Usage: python ingest.py
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def load_documents() -> list[Document]:
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs: list[Document] = []

    for pdf_path in data_dir.glob("*.pdf"):
        pages = PyPDFLoader(str(pdf_path)).load()
        for p in pages:
            p.metadata = p.metadata or {}
            p.metadata["source"] = str(pdf_path)
            p.metadata["filename"] = pdf_path.name
            docs.append(p)

    for txt_path in data_dir.glob("*.txt"):
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(txt_path),
                    "filename": txt_path.name,
                    "page": 1,
                },
            )
        )

    for md_path in data_dir.glob("*.md"):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(md_path),
                    "filename": md_path.name,
                    "page": 1,
                },
            )
        )

    if not docs:
        raise ValueError(f"No documents found in {data_dir}")

    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        md = chunk.metadata or {}
        source = md.get("source", "unknown")
        page = md.get("page", "?")
        chunk.metadata["chunk_id"] = f"{Path(str(source)).name}:{page}:{i}"

    return chunks


def serialize_chunks(chunks: list[Document]) -> list[dict]:
    payload = []
    for ch in chunks:
        md = ch.metadata or {}
        payload.append(
            {
                "chunk_id": md.get("chunk_id"),
                "text": ch.page_content,
                "source": md.get("source", "unknown"),
                "filename": md.get("filename", "unknown"),
                "page": md.get("page", "?"),
            }
        )
    return payload


def build_vectorstore(chunks: list[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    return FAISS.from_documents(chunks, embeddings)


def save_artifacts(vectorstore: FAISS, chunks_payload: list[dict]) -> None:
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    try:
        vectorstore.save_local(str(index_dir))
    except RuntimeError as exc:
        raise RuntimeError(
            f"FAISS could not write to '{index_dir}'. On Windows this is often caused by OneDrive or non-ASCII path characters. Current resolved index_dir: '{settings.index_dir}'. Original error: {exc}"
        ) from exc

    chunks_path = Path(settings.chunks_path)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_payload, f, ensure_ascii=False, indent=2)


def ingest() -> None:
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    chunks_payload = serialize_chunks(chunks)
    save_artifacts(vectorstore, chunks_payload)

    print(f"Loaded docs: {len(docs)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Saved FAISS index to: {settings.index_dir}")
    print(f"Saved chunks metadata to: {settings.chunks_path}")


if __name__ == "__main__":
    ingest()