"""Retrieval evaluation framework using domain-labeled ground-truth QA pairs."""
from typing import List, Dict, Any


class RetrievalEvaluator:
    """
    Evaluates retrieval quality by running ground-truth questions through the
    RAGAssistant and measuring how well the retrieved chunks match the expected domain.

    Metrics produced:
    - source_accuracy   : fraction of queries where the expected domain appears
                          anywhere in the top-k retrieved chunks.
    - top1_accuracy     : fraction of queries where the expected domain is the
                          top-ranked retrieved chunk.
    - mean_top_similarity: average cosine similarity of the highest-ranked chunk
                          across all queries.

    Usage:
        evaluator = RetrievalEvaluator()
        output = evaluator.run(assistant, n_results=3)
        RetrievalEvaluator.print_report(output)
    """

    GROUND_TRUTH: List[Dict[str, str]] = [
        {"question": "What is machine learning?",                          "expected_domain": "AI"},
        {"question": "How does deep learning work?",                       "expected_domain": "AI"},
        {"question": "What are key AI ethics concerns?",                   "expected_domain": "AI"},
        {"question": "What is quantum entanglement?",                      "expected_domain": "Quantum Computing"},
        {"question": "How do quantum computers differ from classical computers?", "expected_domain": "Quantum Computing"},
        {"question": "How does CRISPR gene editing work?",                 "expected_domain": "Biotechnology"},
        {"question": "What are applications of genomics in medicine?",     "expected_domain": "Biotechnology"},
        {"question": "What causes climate change?",                        "expected_domain": "Climate Science"},
        {"question": "What are the effects of greenhouse gas emissions?",  "expected_domain": "Climate Science"},
        {"question": "How does solar energy generate electricity?",        "expected_domain": "Sustainable Energy"},
        {"question": "What are the benefits of wind power?",               "expected_domain": "Sustainable Energy"},
        {"question": "What was the Apollo moon mission?",                  "expected_domain": "Space Exploration"},
        {"question": "How do rockets achieve orbit?",                      "expected_domain": "Space Exploration"},
    ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, assistant, n_results: int = 3) -> Dict[str, Any]:
        """
        Run the full evaluation suite against a RAGAssistant instance.

        Args:
            assistant: An initialised RAGAssistant with documents already ingested.
            n_results: Number of chunks to retrieve per query (top-k).

        Returns:
            Dict with 'results' (per-query detail) and 'metrics' (aggregate scores).
        """
        results = []
        for item in self.GROUND_TRUTH:
            question = item["question"]
            expected_domain = item["expected_domain"]

            response = assistant.query(question, n_results=n_results)

            retrieved_domains = self._get_domains(response)
            top_domain = retrieved_domains[0] if retrieved_domains else None
            domain_hit = expected_domain in retrieved_domains
            top_similarity = self._top_similarity(response)

            results.append({
                "question": question,
                "expected_domain": expected_domain,
                "retrieved_domains": retrieved_domains,
                "top_domain": top_domain,
                "domain_hit": domain_hit,
                "top_similarity": top_similarity,
            })

        metrics = self._aggregate(results)
        return {"results": results, "metrics": metrics}

    @staticmethod
    def print_report(eval_output: Dict[str, Any]) -> None:
        """Print a formatted evaluation report to stdout."""
        results = eval_output["results"]
        metrics = eval_output["metrics"]

        print("\n" + "=" * 72)
        print("  RETRIEVAL EVALUATION REPORT")
        print("=" * 72)
        for r in results:
            status = "PASS" if r["domain_hit"] else "FAIL"
            domains_str = ", ".join(r["retrieved_domains"]) or "none"
            print(f"[{status}] {r['question']}")
            print(f"       Expected : {r['expected_domain']}")
            print(f"       Retrieved: {domains_str}")
            print(f"       Top sim  : {r['top_similarity']:.4f}\n")

        print("-" * 72)
        print(f"  Source Accuracy  (domain in top-k)  : {metrics['source_accuracy']:.2%}")
        print(f"  Top-1 Accuracy   (domain ranked #1) : {metrics['top1_accuracy']:.2%}")
        print(f"  Mean Top Similarity                 : {metrics['mean_top_similarity']:.4f}")
        print(f"  Total Queries Evaluated             : {metrics['total_queries']}")
        print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_domains(self, response: Dict[str, Any]) -> List[str]:
        """Extract ordered, deduplicated domain labels from retrieved metadata."""
        metadatas = response.get("metadatas", [])
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        seen: set = set()
        ordered: List[str] = []
        for m in metadatas:
            domain = m.get("domain") or m.get("source", "Unknown")
            if domain not in seen:
                seen.add(domain)
                ordered.append(domain)
        return ordered

    def _top_similarity(self, response: Dict[str, Any]) -> float:
        """Return cosine similarity of the highest-ranked retrieved chunk."""
        distances = response.get("distances", [])
        if distances and isinstance(distances[0], list):
            distances = distances[0]
        if distances:
            # Both VectorDB and ChromaDB store cosine *distance* (1 - similarity).
            return round(1.0 - float(distances[0]), 4)
        return 0.0

    def _aggregate(self, results: List[Dict]) -> Dict[str, Any]:
        n = len(results)
        if n == 0:
            return {}
        source_acc = sum(1 for r in results if r["domain_hit"]) / n
        top1_acc = sum(1 for r in results if r["top_domain"] == r["expected_domain"]) / n
        mean_sim = sum(r["top_similarity"] for r in results) / n
        return {
            "total_queries": n,
            "source_accuracy": round(source_acc, 4),
            "top1_accuracy": round(top1_acc, 4),
            "mean_top_similarity": round(mean_sim, 4),
        }
