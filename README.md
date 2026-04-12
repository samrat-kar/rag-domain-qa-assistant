# An In-Memory OpenAI-Embedded RAG Domain QnA Assistant

This project implements a **Retrieval-Augmented Generation (RAG)** assistant that answers questions using a custom local document set. The core deliverable is a working pipeline that performs document ingestion, chunking, embedding, retrieval, and grounded response generation with OpenAI models.

## Links

- Publication: [Practical RAG for Domain QA: An In-Memory, OpenAI-Embedded Retrieval-Augmented Generation Assistant](https://app.readytensor.ai/publications/practical-rag-for-domain-qa-an-in-memory-openai-embedded-retrieval-augmented-generation-assistant-TIgCxjV1jspX)

---

## 1) Purpose and Objectives

### Purpose
Build and evaluate a simple, reproducible RAG assistant that can answer user questions based only on provided documents in the data folder.

### Specific Objectives
1. Ingest local text documents from a custom corpus.
2. Split documents into retrieval-friendly chunks.
3. Create vector embeddings for chunks.
4. Retrieve top-k relevant chunks for each query.
5. Generate context-grounded answers with an LLM.
6. Provide a minimal CLI interface for interactive testing.
7. Return answer sources for transparency.

## 2) Intended Audience and Use Case

### Target Audience
- Students learning RAG fundamentals.
- Early-stage AI practitioners building document QA systems.
- Developers who need a small, understandable baseline before production-scale deployment.

### Primary Use Case
Ask domain questions over a curated document collection and get grounded answers with source references.

### Prerequisites
- Python 3.10+ (tested in this workspace with Python 3.14).
- Basic Python and command-line familiarity.
- OpenAI API key.

## 3) Problem Definition

Generic LLM responses can be inaccurate for niche knowledge domains or private documents. This project addresses that limitation by retrieving relevant context from user-provided files before generation, reducing hallucination risk and increasing answer relevance for the selected corpus.

## 4) Scope, Assumptions, and Boundaries

### In Scope
- Text-based document QA over local files.
- Retrieval using dense embeddings and cosine similarity.
- CLI-based interaction.

### Out of Scope
- Web app/UI deployment.
- Multi-user session management.
- Fine-tuning custom LLM weights.
- Large-scale distributed indexing.

### Assumptions
- Input files are valid UTF-8 text.
- The OpenAI API key is available at runtime.
- Document volume is small-to-medium and fits memory.

## 5) Dataset Sources and Characteristics

### Data Source
All documents are local files in the data directory:
- artificial_intelligence.txt
- biotechnology.txt
- climate_science.txt
- quantum_computing.txt
- sample_documents.txt
- space_exploration.txt
- sustainable_energy.txt

### Basic Dataset Stats
- Number of source documents: 7
- File format: `.txt`
- Domain coverage: AI, biotech, climate, quantum, space, sustainability

## 6) Methodology

The implemented pipeline follows a standard RAG flow:

1. **Load documents**  
    Read supported files from the data directory.

2. **Chunk documents**  
    Use `RecursiveCharacterTextSplitter` with overlap to preserve local context.

3. **Embed chunks**  
    Generate OpenAI embeddings (`text-embedding-3-small` by default).

4. **Index in memory**  
    Store chunk text, metadata, IDs, and vectors.

5. **Retrieve top-k chunks**  
    Compute cosine similarity between query vector and indexed vectors.

6. **Prompt + generate**  
    Build a grounded prompt with retrieved context and generate an answer with `gpt-4o-mini` (default).

7. **Return answer + sources**  
    Output answer text, context chunks, and source filenames.

## 7) Implementation Details

### Main Components
- src/app.py  
  `RAGAssistant` orchestration (`load_documents()`, `load_and_ingest()`, `query()`).

- src/vectordb.py  
  In-memory vector index (`chunk_text()`, `add_documents()`, `search()`).

- demo.py  
  Minimal CLI for sample queries and interactive Q&A.

### Tools/Frameworks
- LangChain core abstractions (prompting/output parsing)
- LangChain OpenAI integrations
- OpenAI embeddings and chat model
- NumPy for vector math
- Python dotenv for environment config

### VectorDB Design and Retrieval Logic
This `VectorDB` is a simple in-memory vector store that enables semantic retrieval for a RAG system. During ingestion, each document is split into smaller chunks using a recursive character-based text splitter. Chunking improves retrieval precision by ensuring embeddings represent focused, semantically coherent text instead of entire large documents. A small chunk overlap is used to preserve context across boundaries so important information isn’t lost between adjacent segments.

Each chunk is converted into a dense vector using OpenAI embeddings and stored in a NumPy matrix along with its metadata. At query time, the user question is embedded into the same vector space. The system then computes cosine similarity between the query vector and all stored document vectors. Cosine similarity is used because it measures semantic direction similarity independent of vector magnitude, making it well-suited for text embeddings.

The top-k most similar chunks are returned and passed to the LLM as grounded context, enabling accurate Retrieval-Augmented Generation.

## 8) Evaluation Framework and Verification

### Evaluation Strategy
This submission uses functional validation and qualitative retrieval checks:

1. **Ingestion verification**: confirm all files load and chunk successfully.
2. **Retrieval verification**: confirm top-k chunks are semantically relevant to test queries.
3. **Grounding verification**: confirm generated answers align with retrieved context.
4. **Source transparency**: confirm source filenames are returned with responses.

### Example Verification Queries
- “What is machine learning?”
- “How does deep learning work?”
- “What are key AI ethics concerns?”

## 9) Results Summary

Observed behavior in local runs:
- Documents are loaded and indexed successfully.
- Retrieval returns relevant chunks for sample domain questions.
- Answers are coherent and grounded in the corpus.
- Source file names are returned for traceability.

## 10) Limitations and Trade-offs

1. **In-memory index only**  
    Not designed for very large corpora or persistent multi-session serving.

2. **No reranker**  
    Retrieval quality depends on embedding similarity alone.

3. **No quantitative benchmark**  
    Evaluation is currently functional + qualitative, not a full benchmark suite.

4. **API dependency**  
    Requires OpenAI access at runtime.

## 11) Deployment and Maintenance Considerations

### Deployment
- Suitable for local execution as a CLI tool.
- For production: add persistent vector storage, caching, auth, monitoring, and API rate-limit handling.

### Monitoring Suggestions
- Log query latency and retrieval latency.
- Track top-k source distribution.
- Track user feedback on answer usefulness.

### Maintenance
- Refresh corpus periodically.
- Re-index after significant data updates.
- Review prompt and retrieval settings based on observed failure cases.

## 12) Reproducibility and Setup

### Environment Variables
Create `.env` with:

OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

### Install
Use the project environment and install dependencies from requirements file.

### Run
From project root:

python demo.py

Then test predefined and interactive questions.

## 13) Why This Work Matters

This project provides a compact, understandable RAG baseline for document-grounded QA. It is practical for learning and small-domain assistant prototypes, and it establishes a clean foundation for future extensions (persistent vector DB, reranking, quantitative evaluation, and service deployment).

---

## License

Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — free for non-commercial use with attribution.

---

## Contact

**Maintainer:** Samrat Kar  
**GitHub:** [samrat-kar](https://github.com/samrat-kar)
