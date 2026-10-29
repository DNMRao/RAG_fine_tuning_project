"""Choose the abstention threshold from calibration.json only (never cases.json)."""

import json
from pathlib import Path

from src.ingestion import build_chunks, load_documents
#from src.retrieval import MIN_SCORE, TOP_K, Retriever

from src.config import MIN_SCORE, TOP_K 

from src.retrieval import Retriever

CALIBRATION_PATH = Path("data/calibration/calibration.json")


def load_cases(path: Path = CALIBRATION_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    cases = load_cases()
    retriever = Retriever(build_chunks(load_documents()))

    answer_scores, abstain_scores = [], []
    print(f"{'id':6} {'expected':8} {'top':>6}  top-{TOP_K} hits")
    for case in cases:
        hits = retriever.search(case["question"], k=TOP_K)
        top = hits[0].score
        label = "abstain" if case["should_abstain"] else "answer"
        (abstain_scores if case["should_abstain"] else answer_scores).append(top)
        hit_text = " | ".join(f"{h.score:.3f} {h.chunk.chunk_id}" for h in hits)
        print(f"{case['id']:6} {label:8} {top:6.3f}  {hit_text}")

    hi_abstain, lo_answer = max(abstain_scores), min(answer_scores)
    print(f"\nHighest top score among abstain cases: {hi_abstain:.3f}")
    print(f"Lowest top score among answer cases:   {lo_answer:.3f}")
    if hi_abstain < lo_answer:
        print(f"Separable. Any threshold in ({hi_abstain:.3f}, {lo_answer:.3f}] works; "
              f"midpoint = {(hi_abstain + lo_answer) / 2:.3f}")
    else:
        print("Not separable by top score alone: some abstain case outscores an answer case.")

    # Decision accuracy of the frozen threshold on this set.
    correct = sum(
        (retriever.search(c["question"], k=TOP_K)[0].score < MIN_SCORE) == c["should_abstain"]
        for c in cases
    )
    print(f"\nFrozen MIN_SCORE = {MIN_SCORE}: {correct}/{len(cases)} calibration decisions correct")

# Sweep candidate thresholds: evidence for the choice of MIN_SCORE.
    tops = [(retriever.search(c["question"], k=1)[0].score, c["should_abstain"]) for c in cases]
    print("\nThreshold sweep:")
    for t in [0.30, 0.40, 0.45, 0.49, 0.55, 0.60]:
        correct = sum((score < t) == abstain for score, abstain in tops)
        print(f"  MIN_SCORE={t:.2f}: {correct}/{len(tops)} correct")


if __name__ == "__main__":
    main()

