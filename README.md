# sie-rag

Retrieval-augmented answers over a local corpus, on three small SIE models.

This is the retrieval half of a personal research assistant: embed the corpus,
rerank the shortlist, then answer with citations. The same three calls are what
I'd point at a few thousand PDFs and notes. Here it runs on a handful of markdown
files so the whole pipeline is runnable in one command.

## How it works

| Step | Model | Endpoint |
| --- | --- | --- |
| Embed the corpus and the question | `BAAI/bge-m3` | `/v1/encode` |
| Rerank the shortlist | `Qwen/Qwen3-Reranker-0.6B` | `/v1/score` |
| Write the answer with citations | `Qwen/Qwen3.8-27B-FP8` | `/v1/chat/completions` |

Embeddings pick the candidates; the reranker puts the best passage first so a weak
retriever does not poison the answer. Only the top passages reach the chat model.

## Why SIE

One API for the encoder, the reranker and the chat model. The same three calls run
against SIE Serverless or a self-hosted SIE Cluster, so the pipeline does not change
when the corpus grows from a handful of files to the real pile.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your sk-sie- key in .env
```

## Run

```bash
python ask.py "How do I rotate an API key, and are audit logs included in the free plan?"
```

Add more `.md` files to `corpus/` and they are picked up automatically.

## Example output

```
$ python ask.py "How do I rotate an API key, and are audit logs included in the free plan?"
retrieved: api.md 0.635, pricing.md 0.531, security.md 0.503
reranked:  api.md 0.999, pricing.md 0.954

To rotate an API key, you must first create a replacement key, deploy it to your
clients, and then revoke the old key [api.md]. Note that there is no automatic
overlap window, but two keys can be valid simultaneously during the migration
process [api.md].

Audit logs are not included in the Free plan; they are exclusive to the Business
plan [pricing.md].
```

## Notes

- Retrieval is brute-force cosine over in-memory vectors. On a real corpus you would
  swap in a vector store; the `/v1/encode` and `/v1/score` calls stay the same.
- `RETRIEVE_K` and `RERANK_K` at the top of `ask.py` control how many candidates
  reach the reranker and the chat model.
- Next step: put the scanned PDFs through LightOnOCR-3 on SIE to get Markdown, so
  the same embed / rerank / answer pipeline covers the whole pile, not just the
  `.md` files.
- Point `SIE_BASE_URL` at a self-hosted SIE Cluster and the same three calls run
  there, which is the plan once the volume justifies it.

## License

MIT
