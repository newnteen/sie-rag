"""Answer questions over a small local corpus with a three-step SIE pipeline.

  1. embed the corpus and the question      BAAI/bge-m3               (/v1/encode)
  2. rerank the shortlist                   Qwen/Qwen3-Reranker-0.6B  (/v1/score)
  3. write the answer with citations        Qwen/Qwen3.8-27B-FP8      (/v1/chat/completions)

Everything runs against one API, so the same code moves to a self-hosted SIE
Cluster by pointing SIE_BASE_URL somewhere else.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("SIE_BASE_URL", "https://api.superlinked.com")
API_KEY = os.environ.get("SIE_API_KEY")
if not API_KEY:
    raise SystemExit(
        "SIE_API_KEY is not set. Copy .env.example to .env and put your sk-sie- key in it."
    )

EMBED_MODEL = os.environ.get("SIE_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("SIE_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
CHAT_MODEL = os.environ.get("SIE_CHAT_MODEL", "Qwen/Qwen3.8-27B-FP8")

RETRIEVE_K = 3
RERANK_K = 2


def _post(path: str, payload: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        resp = requests.post(
            f"{BASE_URL}{path}",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                # SIE defaults to msgpack; ask for JSON so resp.json() works.
                "Accept": "application/json",
            },
            json=payload,
            timeout=120,
        )
        # Serverless returns 503 while a model cold-starts. Back off and retry.
        if resp.status_code in (429, 503) and attempt < retries - 1:
            try:
                delay = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                delay = 2**attempt
            time.sleep(min(delay, 30))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")  # loop always returns or raises


def load_corpus(folder: pathlib.Path) -> list[tuple[str, str]]:
    return [
        (p.name, p.read_text(encoding="utf-8"))
        for p in sorted(folder.glob("*.md"))
    ]


def embed(texts: list[str]) -> list[list[float]]:
    out = _post(f"/v1/encode/{EMBED_MODEL}", {"items": [{"text": t} for t in texts]})
    return [item["dense"]["values"] for item in out["items"]]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(question: str, corpus: list[tuple[str, str]]) -> list[tuple[str, str, float]]:
    vectors = embed([question] + [text for _, text in corpus])
    q_vec, doc_vecs = vectors[0], vectors[1:]
    scored = [
        (name, text, cosine(q_vec, vec))
        for (name, text), vec in zip(corpus, doc_vecs)
    ]
    scored.sort(key=lambda row: -row[2])
    return scored[:RETRIEVE_K]


def rerank(question: str, candidates: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    by_id = {name: text for name, text, _ in candidates}
    out = _post(
        f"/v1/score/{RERANK_MODEL}",
        {
            "query": {"text": question},
            "items": [{"id": name, "text": text} for name, text, _ in candidates],
        },
    )
    ranked = sorted(out["scores"], key=lambda s: s["rank"])
    return [(s["item_id"], by_id[s["item_id"]], float(s["score"])) for s in ranked][:RERANK_K]


def answer(question: str, passages: list[tuple[str, str, float]]) -> str:
    context = "\n\n".join(f"[{name}]\n{text}" for name, text, _ in passages)
    out = _post(
        "/v1/chat/completions",
        {
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the context. Cite the source "
                        "file for each fact in square brackets, like [api.md]. If the "
                        "context does not contain the answer, say so."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "max_completion_tokens": 400,
            "temperature": 0,
            "presence_penalty": 0,
        },
    )
    return out["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--corpus", type=pathlib.Path, default=pathlib.Path("corpus"))
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    if not corpus:
        raise SystemExit(f"no markdown files found in {args.corpus}")

    shortlist = retrieve(args.question, corpus)
    print("retrieved:", ", ".join(f"{name} {score:.3f}" for name, _, score in shortlist))

    passages = rerank(args.question, shortlist)
    print("reranked: ", ", ".join(f"{name} {score:.3f}" for name, _, score in passages))

    print("\n" + answer(args.question, passages))


if __name__ == "__main__":
    main()
