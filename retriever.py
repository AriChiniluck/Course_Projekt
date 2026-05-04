import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import settings


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(
        self,
        vectorstore: FAISS,
        chunks: list[dict[str, Any]],
        bm25: BM25Okapi,
        reranker: CrossEncoder,
    ) -> None:
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.bm25 = bm25
        self.reranker = reranker

    @classmethod
    def load(cls) -> "HybridRetriever":
        index_path = Path(settings.index_dir)
        chunks_path = Path(settings.chunks_path)

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError("Knowledge index not found. Run: python ingest.py")

        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )

        vectorstore = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        tokenized_corpus = [tokenize(c.get("text", "")) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        reranker = CrossEncoder(settings.reranker_model)

        return cls(
            vectorstore=vectorstore,
            chunks=chunks,
            bm25=bm25,
            reranker=reranker,
        )

    def semantic_search(self, query: str) -> list[dict[str, Any]]:
        pairs = self.vectorstore.similarity_search_with_score(
            query, k=settings.semantic_top_k
        )

        normalized = []
        for rank, (doc, score) in enumerate(pairs):
            md = doc.metadata or {}
            normalized.append(
                {
                    "chunk_id": md.get("chunk_id"),
                    "text": doc.page_content,
                    "source": md.get("source", "unknown"),
                    "filename": md.get("filename", "unknown"),
                    "page": md.get("page", "?"),
                    "semantic_score": float(score),
                    "semantic_rank": rank,
                }
            )
        return normalized

    def bm25_search(self, query: str) -> list[dict[str, Any]]:
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[: settings.bm25_top_k]

        out = []
        for rank, idx in enumerate(top_idx):
            ch = self.chunks[idx]
            out.append(
                {
                    "chunk_id": ch.get("chunk_id"),
                    "text": ch.get("text", ""),
                    "source": ch.get("source", "unknown"),
                    "filename": ch.get("filename", "unknown"),
                    "page": ch.get("page", "?"),
                    "bm25_score": float(scores[idx]),
                    "bm25_rank": rank,
                }
            )
        return out

    def merge_results(
        self,
        semantic_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        for item in semantic_results:
            key = item.get("chunk_id") or item.get("text", "")[:120]
            cur = merged.setdefault(key, {})
            cur.update(item)
            cur["semantic_rank_score"] = 1.0 / (item["semantic_rank"] + 1)

        for item in bm25_results:
            key = item.get("chunk_id") or item.get("text", "")[:120]
            cur = merged.setdefault(key, {})
            cur.update(item)
            cur["bm25_rank_score"] = 1.0 / (item["bm25_rank"] + 1)

        final = []
        for _, item in merged.items():
            s = item.get("semantic_rank_score", 0.0)
            b = item.get("bm25_rank_score", 0.0)
            item["hybrid_score"] = s + b
            final.append(item)

        final.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return final[: settings.hybrid_top_k]

    def rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[: settings.rerank_top_n]

    def search(self, query: str) -> list[dict[str, Any]]:
        semantic = self.semantic_search(query)
        bm25 = self.bm25_search(query)
        hybrid = self.merge_results(semantic, bm25)
        return self.rerank(query, hybrid)


_RETRIEVER: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever.load()
    return _RETRIEVER
