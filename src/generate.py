import re
import sys
import time

import requests

from retrieve import DEFAULT_INDEX, retrieve

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_LLM = "llama3.2:3b"
REFUSAL = "I don't know based on the papers."
SYSTEM = (
    "You answer questions about research papers using only the sources given. "
    "Write the answer first, in one or two sentences, with exact numbers when the sources give them. "
    "Then cite the sources you used by their labels, like [S1] or [S2][S3]. "
    "Never cite plain bracketed numbers such as [5]; those are the papers' own references. "
    f"If the sources do not contain the answer, reply exactly: {REFUSAL}"
)


def build_prompt(question, hits):
    context = "\n\n".join(f"Source S{n} — {h['paper']}, page {h['page']}\n{h['text']}" for n, h in enumerate(hits, start=1))
    return f"{context}\n\nQuestion: {question}\nAnswer:"


def call_ollama(messages, llm=DEFAULT_LLM):
    payload = {"model": llm, "messages": messages, "stream": False, "keep_alive": "30m", "options": {"temperature": 0, "num_ctx": 4096}}
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def is_refusal(text):
    return "don't know" in text.lower()


def cited_sources(text, hits):
    if is_refusal(text):
        return []
    numbers = {int(n) for n in re.findall(r"\bS(\d+)\b", text)}
    used = [hits[n - 1] for n in sorted(numbers) if 1 <= n <= len(hits)] or hits
    sources, seen = [], set()
    for h in used:
        if (h["paper"], h["page"]) not in seen:
            seen.add((h["paper"], h["page"]))
            sources.append({"paper": h["paper"], "page": h["page"], "score": round(h["score"], 3)})
    return sources


def answer(question, k=4, index_dir=DEFAULT_INDEX, llm=DEFAULT_LLM):
    hits = retrieve(question, k, index_dir)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(question, hits)}]
    text = call_ollama(messages, llm)
    return {"answer": text, "sources": cited_sources(text, hits)}


if __name__ == "__main__":
    start = time.perf_counter()
    result = answer(" ".join(sys.argv[1:]))
    print(result["answer"], "\n")
    for s in result["sources"]:
        print(f"  - {s['paper']}, page {s['page']} (score {s['score']})")
    print(f"\n{time.perf_counter() - start:.1f}s")
