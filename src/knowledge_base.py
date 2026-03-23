"""Persistent knowledge base backed by ChromaDB with domain-aware metadata."""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vectordb import DOMAIN_MAP, extract_domain


class ChromaKnowledgeBase:
    """
    Persistent knowledge base using ChromaDB as the vector store.

    Advantages over the in-memory VectorDB:
    - Documents are persisted to disk; no re-embedding on every startup.
    - Incremental indexing: already-indexed sources are skipped automatically.
    - Domain-aware filtering is handled natively via ChromaDB metadata queries.

    Usage:
        kb = ChromaKnowledgeBase(persist_dir="./chroma_db")
        kb.add_documents(documents)          # skips already-indexed sources
        results = kb.search("What is ML?")
        results = kb.search("What is ML?", domain_filter="AI")
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = None,
        embedding_model: str = None,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")

        self.embedding_model_name = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.embedding_model = OpenAIEmbeddings(
            model=self.embedding_model_name, api_key=api_key
        )

        self._client = chromadb.PersistentClient(path=persist_dir)
        collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        # Use cosine distance so similarity scores are comparable with VectorDB.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"ChromaDB knowledge base ready: '{collection_name}' "
            f"({self._collection.count()} chunks indexed)"
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        skip_existing: bool = True,
    ) -> None:
        """
        Chunk, embed, and index documents into ChromaDB.

        Args:
            documents: List of dicts with 'content' and 'metadata' keys.
            chunk_size: Characters per chunk.
            chunk_overlap: Overlap between adjacent chunks.
            skip_existing: If True, sources already in the collection are skipped.
        """
        existing_sources: set = set()
        if skip_existing and self._collection.count() > 0:
            all_meta = self._collection.get(include=["metadatas"])["metadatas"]
            existing_sources = {m.get("source", "") for m in all_meta if m}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks: List[str] = []
        all_ids: List[str] = []
        all_metadatas: List[Dict[str, str]] = []

        print(f"Processing {len(documents)} documents...")
        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            source = metadata.get("source", f"doc_{doc_idx}")

            if source in existing_sources:
                print(f"  Skipping already-indexed: {source}")
                continue

            domain = extract_domain(source)
            chunks = splitter.split_text(content)
            print(f"  {source} ({domain}): {len(chunks)} chunks")

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{source}_chunk_{chunk_idx}")
                all_metadatas.append({
                    "source": source,
                    "domain": domain,
                    "doc_index": str(doc_idx),
                    "chunk_index": str(chunk_idx),
                })

        if not all_chunks:
            print("No new chunks to index.")
            return

        print(f"  Generating embeddings for {len(all_chunks)} new chunks...")
        embeddings = self.embedding_model.embed_documents(all_chunks)
        self._collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            ids=all_ids,
            metadatas=all_metadatas,
        )
        print(
            f"Indexed {len(all_chunks)} new chunks. "
            f"Total in collection: {self._collection.count()}"
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        domain_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve the top-k most similar chunks for a query.

        Args:
            query: Natural language question.
            n_results: Number of chunks to return.
            domain_filter: Restrict results to a specific domain label
                           (e.g. "AI", "Biotechnology"). Use list_domains() to see options.

        Returns:
            Dict with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        count = self._collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        query_embedding = self.embedding_model.embed_query(query)
        where = {"domain": {"$eq": domain_filter}} if domain_filter else None

        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_domains(self) -> List[str]:
        """Return sorted list of distinct domain labels in the collection."""
        if self._collection.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("domain", "Unknown") for m in all_meta if m})

    def clear(self) -> None:
        """Remove all documents from the collection."""
        if self._collection.count() == 0:
            print("Collection is already empty.")
            return
        ids = self._collection.get()["ids"]
        self._collection.delete(ids=ids)
        print(f"Cleared {len(ids)} chunks from the collection.")
