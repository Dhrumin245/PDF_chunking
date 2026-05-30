# PDF RAG Token Optimization POC

Proof of concept for PDF chunking and RAG retrieval optimized for minimal token consumption.

The implementation follows this layout:

```text
pdf-rag-token-optimization-poc/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── sample.pdf
├── src/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── token_tracker.py
│   └── rag_pipeline.py
├── experiments/
│   ├── chunk_size_128.json
│   ├── chunk_size_256.json
│   └── chunk_size_512.json
├── results/
│   ├── metrics.csv
│   └── retrieval_comparison.md
├── tests/
│   ├── test_chunking.py
│   └── test_retrieval.py
└── app.py
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python app.py --pdf data/sample.pdf --query "What is the leave policy?" --chunk-size 256
```

Lower-token retrieval:

```powershell
python app.py `
  --pdf data/sample.pdf `
  --query "What is the leave policy?" `
  --chunk-size 128 `
  --top-k 4 `
  --context-budget 600
```

## Module Responsibilities

- `src/pdf_loader.py`: `extract_text(pdf_path)` extracts raw PDF text.
- `src/chunker.py`: `chunk_text(text, chunk_size)` creates token-bounded chunks.
- `src/embeddings.py`: `generate_embeddings(chunks)` creates deterministic local hashed embeddings.
- `src/vector_store.py`: `build_faiss_index()` uses FAISS if installed, otherwise a pure Python vector store.
- `src/retriever.py`: `retrieve_top_k()` retrieves and packs only relevant chunks under a context token budget.
- `src/token_tracker.py`: `count_tokens()` and `TokenTracker` report prompt/query/context token usage.
- `src/rag_pipeline.py`: orchestrates loading, chunking, embedding, indexing, retrieval, prompt building, and metrics.

## Token Optimization Defaults

- Chunk sizes are constrained to `128`, `256`, or `512` tokens.
- Default overlap is `10%` of chunk size, capped at `48` tokens.
- Retrieval first ranks by vector similarity, then packs chunks until `context_budget` is reached.
- The final output reports query tokens, context tokens, prompt tokens, answer budget, and estimated total tokens.

## Experiments

The `experiments/` directory stores repeatable settings for chunk sizes `128`, `256`, and `512`.

The `results/` directory includes a starter metrics table and retrieval comparison notes. Run the same query across experiment configs and compare:

- chunks created
- chunks retrieved
- context tokens
- prompt tokens
- estimated total tokens
- qualitative relevance

## Tests

```powershell
python -m unittest discover -s tests
```
