import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import DEFAULT_LLM, answer, is_refusal
from ingest import MODELS, build
from retrieve import retrieve

RESULTS = ROOT / "eval" / "results"
CHUNK_SIZES = [300, 500, 800]
K = 4
COLORS = {"bge-small": "#2a78d6", "minilm": "#eb6834"}


def load_questions():
    return json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))


def index_dir(config):
    path = ROOT / "index" / config
    if not (path / "faiss.index").exists():
        model, size = config.rsplit("_", 1)
        build(model, int(size), int(size) // 5, path)
    return path


def is_relevant(item, q):
    return item["paper"] == q["paper"] and item["page"] in q["pages"]


def truncated_share(path):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    chunks = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
    return len(chunks), sum(c["tokens"] + 2 > config["max_seq_length"] for c in chunks) / len(chunks)


def eval_retrieval(questions):
    rows, stats = [], []
    answerable = [q for q in questions if q["type"] != "refusal"]
    for model in MODELS:
        for size in CHUNK_SIZES:
            config = f"{model}_{size}"
            path = index_dir(config)
            n_chunks, truncated = truncated_share(path)
            stats.append({"config": config, "chunks": n_chunks, "truncated": truncated})
            for q in answerable:
                hits = retrieve(q["question"], K, path)
                rank = next((r for r, h in enumerate(hits, start=1) if is_relevant(h, q)), None)
                rows.append({
                    "config": config, "model": model, "chunk_size": size, "id": q["id"], "type": q["type"], "rank": rank,
                    "retrieved": "; ".join(f"{h['paper'].split()[0]} p{h['page']}" for h in hits),
                })
    df = pd.DataFrame(rows)
    df["hit@1"] = df["rank"] == 1
    df["hit@4"] = df["rank"].notna()
    df["rr"] = (1 / df["rank"]).fillna(0)
    df.to_csv(RESULTS / "retrieval.csv", index=False)
    summary = df.groupby(["config", "model", "chunk_size"], sort=False).agg(
        hit1=("hit@1", "mean"), hit4=("hit@4", "mean"), mrr=("rr", "mean")).reset_index()
    table_hit4 = df[df["type"] == "table"].groupby("config")["hit@4"].mean().rename("table_hit4")
    summary = summary.merge(table_hit4, on="config").merge(pd.DataFrame(stats), on="config")
    summary.round(3).to_csv(RESULTS / "retrieval_summary.csv", index=False)
    plot_retrieval(summary)
    return summary


def plot_retrieval(summary):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#898781", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, facecolor="#fcfcfb")
    width = 0.38
    for ax, (metric, title) in zip(axes, [("hit4", "Hit@4"), ("mrr", "MRR")]):
        ax.set_facecolor("#fcfcfb")
        for i, model in enumerate(MODELS):
            rows = summary[summary["model"] == model]
            xs = [CHUNK_SIZES.index(s) + (i - 0.5) * width for s in rows["chunk_size"]]
            ax.bar(xs, rows[metric], width - 0.03, color=COLORS[model], label=model)
        ax.set_title(title, loc="left", color="#0b0b0b", fontsize=11)
        ax.set_xticks(range(len(CHUNK_SIZES)), [f"{s} tokens" for s in CHUNK_SIZES])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    axes[0].legend(frameon=False, loc="upper left", ncols=2, bbox_to_anchor=(0, -0.12))
    fig.suptitle("Retrieval quality by embedding model and chunk size (17 questions, k=4)", x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(RESULTS / "retrieval.png", dpi=160, facecolor=fig.get_facecolor())


def contains_all(text, groups):
    text = text.lower()
    return all(any(alt in text for alt in group.split("|")) for group in groups)


def eval_answers(questions, configs, llm):
    out = RESULTS / "answers.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["config", "llm", "id"])
    finished = set(zip(done["config"], done["llm"], done["id"]))
    for config in configs:
        path = index_dir(config)
        for q in questions:
            if (config, llm, q["id"]) in finished:
                continue
            start = time.perf_counter()
            result = answer(q["question"], K, path, llm)
            refused = is_refusal(result["answer"])
            row = {
                "config": config, "llm": llm, "id": q["id"], "type": q["type"],
                "correct": refused if q["type"] == "refusal" else (not refused and contains_all(result["answer"], q["must_contain"])),
                "refused": refused,
                "cited_correct_page": any(is_relevant(s, q) for s in result["sources"]),
                "seconds": round(time.perf_counter() - start, 1),
                "answer": result["answer"].replace("\n", " "),
                "sources": "; ".join(f"{s['paper'].split()[0]} p{s['page']}" for s in result["sources"]),
            }
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"{config} {llm} {q['id']}: {'OK ' if row['correct'] else 'BAD'} {row['seconds']}s", flush=True)
    summarize_answers(out, questions)


def cites_relevant(sources, q):
    cited = set(sources.split("; ")) if isinstance(sources, str) else set()
    return q["paper"] is not None and any(f"{q['paper'].split()[0]} p{p}" in cited for p in q["pages"])


def summarize_answers(out, questions):
    by_id = {q["id"]: q for q in questions}
    df = pd.read_csv(out)
    df["cited_correct_page"] = [cites_relevant(s, by_id[i]) for s, i in zip(df["sources"], df["id"])]
    df.to_csv(out, index=False)
    answerable, refusal = df[df["type"] != "refusal"], df[df["type"] == "refusal"]
    summary = pd.DataFrame({
        "answer_accuracy": answerable.groupby(["config", "llm"])["correct"].mean(),
        "citation_accuracy": answerable.groupby(["config", "llm"])["cited_correct_page"].mean(),
        "false_refusals": answerable.groupby(["config", "llm"])["refused"].sum(),
        "refusal_accuracy": refusal.groupby(["config", "llm"])["correct"].mean(),
        "mean_seconds": df.groupby(["config", "llm"])["seconds"].mean(),
    }).reset_index()
    summary.round(3).to_csv(RESULTS / "answers_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["retrieval", "answers"])
    parser.add_argument("--configs", nargs="+", default=["bge-small_500"])
    parser.add_argument("--llm", default=DEFAULT_LLM)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    if args.stage == "retrieval":
        print(eval_retrieval(questions).round(3).to_string(index=False))
    else:
        eval_answers(questions, args.configs, args.llm)


if __name__ == "__main__":
    main()
