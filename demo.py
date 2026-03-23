"""CLI demo showcasing RAG retrieval, domain filtering, query rewriting, and evaluation."""
from src.app import RAGAssistant
from src.evaluator import RetrievalEvaluator


def run_example_queries(assistant: RAGAssistant) -> None:
    print("\n" + "=" * 60)
    print("  EXAMPLE QUERIES")
    print("=" * 60)
    questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are key AI ethics concerns?",
    ]
    for q in questions:
        result = assistant.query(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")


def run_domain_filtering(assistant: RAGAssistant) -> None:
    print("\n" + "=" * 60)
    print("  DOMAIN-FILTERED RETRIEVAL")
    print("=" * 60)
    available_domains = assistant.list_domains()
    print(f"Available domains: {', '.join(available_domains)}\n")

    domain_queries = [
        ("How does CRISPR gene editing work?",     "Biotechnology"),
        ("What causes climate change?",            "Climate Science"),
        ("How do quantum computers work?",         "Quantum Computing"),
    ]
    for question, domain in domain_queries:
        result = assistant.query(question, domain_filter=domain)
        print(f"Q [{domain}]: {question}")
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}\n")


def run_query_rewriting(assistant: RAGAssistant) -> None:
    print("\n" + "=" * 60)
    print("  QUERY REWRITING")
    print("=" * 60)

    # Enable query processor for this demo.
    from src.query_processor import QueryProcessor
    assistant.query_processor = QueryProcessor(assistant.llm)

    vague_queries = [
        "Tell me about energy",
        "What about risks in AI?",
        "Compare solar and wind",
    ]
    for q in vague_queries:
        result = assistant.query(q, rewrite_query=True)
        print(f"Original : {q}")
        print(f"Rewritten: {result['retrieval_query']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}\n")

    # Disable query processor after demo.
    assistant.query_processor = None


def run_evaluation(assistant: RAGAssistant) -> None:
    print("\n" + "=" * 60)
    print("  RETRIEVAL EVALUATION")
    print("=" * 60)
    evaluator = RetrievalEvaluator()
    output = evaluator.run(assistant, n_results=3)
    RetrievalEvaluator.print_report(output)


def run_interactive(assistant: RAGAssistant) -> None:
    print("\nInteractive mode (type 'quit' to exit)")
    print("Tip: prefix your question with a domain to filter, e.g.  [AI] What is backpropagation?")
    available = assistant.list_domains()
    domain_tags = {f"[{d}]": d for d in available}

    while True:
        raw = input("\nYou: ").strip()
        if not raw or raw.lower() in {"quit", "exit", "q"}:
            break

        # Parse optional domain tag prefix.
        domain_filter = None
        question = raw
        for tag, domain in domain_tags.items():
            if raw.upper().startswith(tag.upper()):
                domain_filter = domain
                question = raw[len(tag):].strip()
                break

        result = assistant.query(question, domain_filter=domain_filter)
        if domain_filter:
            print(f"(Filtered to domain: {domain_filter})")
        print(f"Assistant: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")


def main():
    assistant = RAGAssistant()
    assistant.load_and_ingest("./data")

    run_example_queries(assistant)
    run_domain_filtering(assistant)
    run_query_rewriting(assistant)
    run_evaluation(assistant)
    run_interactive(assistant)


if __name__ == "__main__":
    main()
