"""Benchmark baseline RAG and hybrid RAG + LoRA on the held-out cases with one shared rubric."""

import json
from pathlib import Path

from src.config import (
    ADAPTER_PATH, BASE_MODEL, BASE_MODEL_REVISION, EMBEDDING_MODEL,
    EMBEDDING_REVISION, MIN_SCORE, TOP_K,
)
from src.data import EvalCase, load_cases
from src.generation import AnswerResult, answer_question, load_model
from src.ingestion import build_chunks, load_documents
from src.retrieval import Retriever

TRACE_PATH = Path("artifacts/evaluation_traces.json")
TABLES_PATH = Path("reports/evaluation_tables.md")
MAX_WORDS = 60
SYSTEMS = {"baseline": False, "hybrid": True}  # name -> use_adapter


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def score_retrieval(case: EvalCase, retriever: Retriever) -> dict:
    """Ranking quality for one case, measured before the abstention gate."""
    hits = retriever.search(case.question, k=TOP_K)
    ranked = [hit.chunk.document_id for hit in hits]
    expected = set(case.expected_sources)
    if not expected:
        return {"top_score": round(hits[0].score, 3), "ranked_sources": ranked,
                "recall_at_k": None, "reciprocal_rank": None}
    rank = next((i + 1 for i, doc in enumerate(ranked) if doc in expected), None)
    return {
        "top_score": round(hits[0].score, 3),
        "ranked_sources": ranked,
        "recall_at_k": len(expected & set(ranked)) / len(expected),
        "reciprocal_rank": 1 / rank if rank else 0.0,
    }


def score_answer(case: EvalCase, result: AnswerResult) -> dict:
    """Apply the deterministic rubric to one system's answer."""
    text = result.raw_output.lower()
    row = {
        "status": result.status,
        "reason": result.reason,
        "output": result.raw_output,
        "citations": result.citations,
        "decision_correct": (result.status == "answered") != case.should_abstain,
        "forbidden_hits": [t for t in case.forbidden_terms if t.lower() in text],
        "latency_seconds": round(result.latency_seconds, 2),
    }
    if not case.should_abstain:
        retrieved_sources = {chunk_id.split("#")[0] for chunk_id in result.retrieved_ids}
        cited = bool(result.citations)
        row["term_coverage"] = mean([t.lower() in text for t in case.required_terms])
        row["citation_faithful"] = cited and set(result.citations) <= retrieved_sources
        row["style_ok"] = cited and len(result.raw_output.split()) <= MAX_WORDS
    return row


def summarize(rows: list[dict]) -> dict:
    supported = [r for r in rows if "term_coverage" in r]
    return {
        "decision_accuracy": mean([r["decision_correct"] for r in rows]),
        "term_coverage": mean([r["term_coverage"] for r in supported]),
        "citation_faithfulness": mean([r["citation_faithful"] for r in supported]),
        "style_compliance": mean([r["style_ok"] for r in supported]),
        "forbidden_claims": sum(len(r["forbidden_hits"]) for r in rows),
        "mean_latency_seconds": mean([r["latency_seconds"] for r in rows]),
        "answered": sum(r["status"] == "answered" for r in rows),
        "abstained": sum(r["status"] == "abstained" for r in rows),
        "rejected": sum(r["status"] == "rejected" for r in rows),
    }


def run_benchmark() -> dict:
    cases = load_cases()
    retriever = Retriever(build_chunks(load_documents()))  # one retriever for both systems
    rows = [
        {"id": c.id, "question": c.question, "should_abstain": c.should_abstain,
         "retrieval": score_retrieval(c, retriever)}
        for c in cases
    ]
    for name, use_adapter in SYSTEMS.items():
        model, tokenizer = load_model(use_adapter)
        for case, row in zip(cases, rows, strict=True):
            row[name] = score_answer(case, answer_question(case.question, retriever, model, tokenizer))

    supported = [r for r in rows if not r["should_abstain"]]
    summary_path = ADAPTER_PATH / "training_summary.json"
    return {
        "config": {
            "embedding_model": f"{EMBEDDING_MODEL}@{EMBEDDING_REVISION}",
            "base_model": f"{BASE_MODEL}@{BASE_MODEL_REVISION}",
            "top_k": TOP_K, "min_score": MIN_SCORE, "max_words": MAX_WORDS,
            "adapter": json.loads(summary_path.read_text(encoding="utf-8")),
        },
        "retrieval": {
            "recall_at_k": mean([r["retrieval"]["recall_at_k"] for r in supported]),
            "mrr": mean([r["retrieval"]["reciprocal_rank"] for r in supported]),
        },
        "systems": {name: summarize([r[name] for r in rows]) for name in SYSTEMS},
        "cases": rows,
    }


def fmt(value) -> str:
    return "-" if value is None else str(round(value, 3)) if isinstance(value, float) else str(value)


def render_markdown(results: dict) -> str:
    """Aggregate and per-case tables for the evaluation report."""
    retrieval = results["retrieval"]
    lines = [
        "# Evaluation Tables", "",
        f"Retrieval on supported cases (k={results['config']['top_k']}): "
        f"recall@k = {fmt(retrieval['recall_at_k'])}, MRR = {fmt(retrieval['mrr'])}", "",
        "| Metric | Baseline RAG | Hybrid RAG + LoRA |", "|---|---|---|",
    ]
    base, hybrid = results["systems"]["baseline"], results["systems"]["hybrid"]
    for key in base:
        lines.append(f"| {key} | {fmt(base[key])} | {fmt(hybrid[key])} |")

    lines += ["", "## Per Case", "",
              "| Case | Expected | Top score | Recall@k | Baseline (coverage) | Hybrid (coverage) |",
              "|---|---|---|---|---|---|"]
    for row in results["cases"]:
        expected = "abstain" if row["should_abstain"] else "answer"
        cells = [f"{row[n]['status']} ({fmt(row[n].get('term_coverage'))})" for n in SYSTEMS]
        lines.append(f"| {row['id']} | {expected} | {fmt(row['retrieval']['top_score'])} | "
                     f"{fmt(row['retrieval']['recall_at_k'])} | {cells[0]} | {cells[1]} |")

    lines += ["", "## Model Outputs", ""]
    for row in results["cases"]:
        lines.append(f"**{row['id']}**: {row['question']}")
        for name in SYSTEMS:
            r = row[name]
            lines.append(f"- {name} — {r['status']} ({r['reason']}): {r['output'] or '-'}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_benchmark()
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    TABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(results)
    TABLES_PATH.write_text(markdown, encoding="utf-8")
    print(markdown.split("## Per Case")[0])
    print(f"Full traces: {TRACE_PATH}\nTables: {TABLES_PATH}")

