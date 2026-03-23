"""RAG assistant with domain-aware retrieval and optional query processing."""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .vectordb import VectorDB
from .query_processor import QueryProcessor

# Load environment variables
load_dotenv()


class RAGAssistant:
    """
    Question-answering assistant powered by Retrieval-Augmented Generation.

    Supports:
    - Domain-aware retrieval with optional domain filtering.
    - LLM-based query rewriting for improved retrieval precision.
    - Pluggable vector store (VectorDB by default; pass a ChromaKnowledgeBase
      instance via the `store` parameter for persistent indexing).
    """

    def __init__(self, store=None, use_query_processor: bool = False):
        """
        Initialise the RAG assistant.

        Args:
            store: Vector store to use. Defaults to an in-memory VectorDB.
                   Pass a ChromaKnowledgeBase instance for persistent storage.
            use_query_processor: If True, rewrite each query with the LLM before
                                 retrieval to improve precision.
        """
        self.llm = self._initialize_llm()

        # Prompt instructs the model to stay strictly within retrieved context.
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Use only the provided context to answer the question.
If the context does not contain enough information, say so honestly rather than making up an answer.

Context:
{context}

Question: {question}

Provide a clear, concise answer based on the context provided."""
        )

        self.chain = self.prompt_template | self.llm | StrOutputParser()

        # Allow injecting a custom store (e.g. ChromaKnowledgeBase).
        self.vector_db = store if store is not None else VectorDB()

        # Optional query processor for rewriting queries before retrieval.
        self.query_processor = QueryProcessor(self.llm) if use_query_processor else None

        print("RAG Assistant initialized successfully")

    # ------------------------------------------------------------------
    # Document loading & ingestion
    # ------------------------------------------------------------------

    def load_documents(self, data_path: str = "./data") -> List[Dict[str, Any]]:
        """
        Load documents from the data directory.

        Args:
            data_path: Path to folder containing .txt / .csv / .json / .md files.

        Returns:
            List of document dicts with 'content' and 'metadata' keys.
        """
        results: List[Dict[str, Any]] = []
        dir_path = Path(data_path)

        if not dir_path.exists():
            print(f"⚠ Data directory not found: {data_path}")
            return results

        for file_path in sorted(dir_path.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in (".txt", ".csv", ".json", ".md"):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                results.append({
                    "content": content,
                    "metadata": {
                        "source": file_path.name,
                        "path": str(file_path),
                    },
                })
            except Exception as exc:
                print(f"⚠ Could not read {file_path.name}: {exc}")

        print(f"Loaded {len(results)} documents from {data_path}")
        return results

    def load_and_ingest(self, data_path: str = "./data") -> None:
        """Load documents from disk and ingest them into the vector store."""
        documents = self.load_documents(data_path)
        if documents:
            self.vector_db.add_documents(documents)
        else:
            print("No documents to ingest.")

    # ------------------------------------------------------------------
    # RAG query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        n_results: int = 3,
        domain_filter: Optional[str] = None,
        rewrite_query: bool = False,
    ) -> Dict[str, Any]:
        """
        Answer a question using retrieved context (RAG).

        Args:
            question: The user's question.
            n_results: Number of context chunks to retrieve.
            domain_filter: Restrict retrieval to a specific domain label
                           (e.g. "AI", "Biotechnology", "Climate Science").
                           Use vector_db.list_domains() to see available options.
            rewrite_query: If True (or if use_query_processor=True was set at init),
                           rewrite the query with the LLM before retrieval.

        Returns:
            Dict with keys: 'question', 'retrieval_query', 'answer',
                            'context_chunks', 'sources', 'metadatas', 'distances'.
        """
        # Optionally rewrite the query for better retrieval alignment.
        retrieval_query = question
        if (rewrite_query or self.query_processor) and self.query_processor:
            retrieval_query = self.query_processor.rewrite(question)
            if retrieval_query != question:
                print(f"Query rewritten: {retrieval_query}")

        # 1. Retrieve relevant chunks (with optional domain filter).
        search_results = self.vector_db.search(
            retrieval_query, n_results=n_results, domain_filter=domain_filter
        )

        chunks = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        distances = search_results.get("distances", [[]])[0]

        # 2. Build context string from retrieved chunks.
        context = "\n\n---\n\n".join(chunks) if chunks else "No relevant context found."

        # 3. Generate answer via LLM chain.
        answer = self.chain.invoke({"context": context, "question": question})

        # 4. Collect unique source filenames for transparency/citations.
        sources = list({m.get("source", "unknown") for m in metadatas}) if metadatas else []

        return {
            "question": question,
            "retrieval_query": retrieval_query,
            "answer": answer,
            "context_chunks": chunks,
            "sources": sources,
            "metadatas": metadatas,
            "distances": distances,
        }

    def list_domains(self) -> List[str]:
        """Return the domain labels available in the current vector store."""
        return self.vector_db.list_domains()

    def _initialize_llm(self):
        """Initialise the chat model from environment configuration."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required in .env")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Using OpenAI model: {model_name}")
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=0.2)


def main():
    """Quick smoke test: ingest docs and answer one sample question."""
    assistant = RAGAssistant()
    assistant.load_and_ingest("./data")
    result = assistant.query("What is machine learning?")
    print(result["answer"])


if __name__ == "__main__":
    main()
