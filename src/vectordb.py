import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_openai import OpenAIEmbeddings

# Maps source filename stems to human-readable domain labels.
DOMAIN_MAP: Dict[str, str] = {
    "artificial_intelligence": "AI",
    "biotechnology": "Biotechnology",
    "climate_science": "Climate Science",
    "quantum_computing": "Quantum Computing",
    "space_exploration": "Space Exploration",
    "sustainable_energy": "Sustainable Energy",
    "sample_documents": "General",
}


def extract_domain(source: str) -> str:
    """Derive a domain label from a source filename."""
    stem = Path(source).stem
    return DOMAIN_MAP.get(stem, stem.replace("_", " ").title())


class VectorDB:
    """
    A simple in-memory vector database wrapper using OpenAI embeddings.
    Supports domain-aware metadata and optional domain filtering at search time.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Logical collection name (kept for compatibility)
            embedding_model: OpenAI embedding model name
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for embeddings. "
                "Please set OPENAI_API_KEY in your .env file."
            )

        # Initialize embedding client used during ingestion and retrieval.
        print(f"Loading OpenAI embedding model: {self.embedding_model_name}")
        self.embedding_model = OpenAIEmbeddings(
            model=self.embedding_model_name,
            api_key=api_key,
        )

        # In-memory index storage (documents + metadata + vectors).
        self._documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._ids: List[str] = []
        self._embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)

        print(f"Vector database initialized (in-memory): {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Split text into smaller chunks using RecursiveCharacterTextSplitter.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Recursive splitter preserves semantic boundaries better than naive fixed slicing.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents (dicts with 'content' and optional 'metadata')
        """
        print(f"Processing {len(documents)} documents...")

        all_chunks = []
        all_ids = []
        all_metadatas = []

        for doc_idx, doc in enumerate(documents):
            # Support either structured dict input or raw text input.
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}

            chunks = self.chunk_text(content)
            print(f"  Document {doc_idx + 1}: split into {len(chunks)} chunks")

            domain = extract_domain(metadata.get("source", f"doc_{doc_idx}"))
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                # Store serializable metadata including the domain label.
                all_metadatas.append({
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "domain": domain,
                    **{k: str(v) for k, v in metadata.items()},
                })

        if all_chunks:
            print(f"  Generating embeddings for {len(all_chunks)} chunks...")
            embeddings = np.array(
                self.embedding_model.embed_documents(all_chunks),
                dtype=np.float32,
            )

            if self._embeddings.size == 0:
                self._embeddings = embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, embeddings])

            self._documents.extend(all_chunks)
            self._metadatas.extend(all_metadatas)
            self._ids.extend(all_ids)

        print(f"Documents added to vector database ({len(all_chunks)} chunks total)")

    def list_domains(self) -> List[str]:
        """Return sorted list of distinct domain labels currently indexed."""
        return sorted({m.get("domain", "Unknown") for m in self._metadatas})

    def search(self, query: str, n_results: int = 5,
               domain_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return
            domain_filter: If set, restrict results to chunks from this domain label
                           (e.g. "AI", "Biotechnology"). Use list_domains() to see options.

        Returns:
            Dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        count = len(self._documents)
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        # Embed user query with same model space as document vectors.
        query_embedding = np.array(
            self.embedding_model.embed_query(query),
            dtype=np.float32,
        )

        # Cosine similarity across all indexed chunks.
        doc_norms = np.linalg.norm(self._embeddings, axis=1) + 1e-12
        query_norm = np.linalg.norm(query_embedding) + 1e-12
        similarities = (self._embeddings @ query_embedding) / (doc_norms * query_norm)

        # Apply domain filter by masking non-matching chunks before ranking.
        if domain_filter:
            mask = np.array(
                [m.get("domain") == domain_filter for m in self._metadatas],
                dtype=bool,
            )
            similarities = np.where(mask, similarities, -2.0)

        # Get top-k indices by similarity (descending), excluding masked entries.
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = [i for i in sorted_indices if similarities[i] > -2.0][:n_results]

        top_docs = [self._documents[i] for i in top_indices]
        top_metas = [self._metadatas[i] for i in top_indices]
        top_ids = [self._ids[i] for i in top_indices]
        # Store as cosine distance (1 - similarity) for consistency with ChromaDB output.
        top_distances = [float(1.0 - similarities[i]) for i in top_indices]

        return {
            "documents": [top_docs],
            "metadatas": [top_metas],
            "distances": [top_distances],
            "ids": [top_ids],
        }
