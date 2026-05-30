# Retrieval Comparison

Use this file to record the same query across each chunk-size experiment.

| Experiment | Chunk Size | Retrieved Chunks | Context Tokens | Prompt Tokens | Relevance Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| chunk_size_128 | 128 | TBD | TBD | TBD | TBD |
| chunk_size_256 | 256 | TBD | TBD | TBD | TBD |
| chunk_size_512 | 512 | TBD | TBD | TBD | TBD |

## Initial Guidance

- `128` token chunks usually minimize context cost but can split related ideas.
- `256` token chunks are a practical default for policy, handbook, and contract PDFs.
- `512` token chunks can improve recall for long explanations but raise prompt cost.
