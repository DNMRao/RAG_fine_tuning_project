# Retrieval Threshold Notes

**Decision:** `MIN_SCORE = 0.45`, with `TOP_K = 3`. The value was chosen on a
self-written calibration set and frozen before the held-out evaluation cases
(`data/evaluation/cases.json`) were run.

## Setup

| Setting | Value |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243` |
| Vector store | Qdrant (in-memory), cosine distance |
| Chunking | One chunk per paragraph, title prepended (16 chunks from 6 policies) |
| Gate rule | Abstain when the **top-1** score is below `MIN_SCORE`; otherwise pass all top-k hits |
| Calibration data | `data/calibration/calibration.json` (7 answerable, 7 unanswerable) |

The calibration questions were written from the policy files, not from
`cases.json`. They avoid the facts that the evaluation cases test. Most
unanswerable questions are *hard negatives*: they reuse policy vocabulary
(subscription, refund, shipping, password) but ask about something the
policies do not cover.

## Top-1 Scores

| ID | Expected | Top score | Top hit |
|---|---|---|---|
| cal-13 | abstain | 0.237 | shipping#0 |
| cal-08 | abstain | 0.244 | subscriptions#0 |
| cal-11 | abstain | 0.340 | account-security#2 |
| cal-12 | abstain | 0.360 | subscriptions#0 |
| cal-10 | abstain | 0.421 | subscriptions#0 |
| cal-02 | answer | 0.487 | refunds#0 |
| cal-14 | abstain | 0.488 | refunds#0 |
| cal-05 | answer | 0.497 | account-security#2 |
| cal-01 | answer | 0.502 | refunds#0 |
| cal-07 | answer | 0.543 | incidents#0 |
| cal-06 | answer | 0.564 | account-security#1 |
| cal-09 | abstain | 0.581 | subscriptions#0 |
| cal-03 | answer | 0.611 | subscriptions#0 |
| cal-04 | answer | 0.735 | shipping#0 |

The two groups **overlap** between 0.487 and 0.581, so no single threshold
separates them perfectly.

## Threshold Sweep

| MIN_SCORE | Answerable answered (/7) | Unanswerable abstained (/7) | Correct (/14) |
|---|---|---|---|
| 0.30 | 7 | 2 | 9 |
| 0.40 | 7 | 4 | 11 |
| **0.45** | **7** | **5** | **12** |
| 0.49 | 6 | 6 | 12 |
| 0.55 | 3 | 6 | 9 |
| 0.60 | 2 | 7 | 9 |

## Why 0.45

1. **Highest accuracy with a safe margin.** 0.45 and 0.49 both score 12/14.
   But 0.49 sits about 0.002 from two calibration points (cal-02, cal-14),
   so a small rewording would flip either one. 0.45 is the midpoint of the
   gap between 0.421 and 0.487, about 0.03 from the nearest point on each side.
2. **Wrongly answering can be recovered later; wrongly abstaining cannot.** A
   false abstention removes the evidence before generation, and nothing later
   can recover it. A false answer still faces the generator's own "approved
   material is insufficient" rule and citation validation. 0.45 answers every
   answerable calibration question.
3. **The remaining errors are not threshold problems.** Catching cal-09 (0.581)
   would require a threshold above 0.581, which would drop 5 of 7 answerable
   questions.

## Failure Cases

- **cal-09:** "Can I extend my subscription for free because the platform was
  unavailable?" It scores 0.581 against `subscriptions#0`. The question is full
  of subscription and outage vocabulary, but no policy covers outage
  compensation.
- **cal-14:** "Can I collect my refund in cash at a physical store?" It scores
  0.488 against `refunds#0`. That chunk covers refunds generally, but not cash
  or in-store refunds.

Both cases show that cosine similarity measures **topic overlap**, not whether
a passage contains the answer.

## Hypothesis for the Held-Out Evaluation

- Off-topic unanswerable questions should score well below 0.45 and abstain at
  the retrieval gate.
- Questions that are near a policy topic but not covered by it may pass the
  retrieval gate. They must then be stopped by the generator. Citation
  validation will not catch them, because the cited source *was* retrieved.
- This is where baseline RAG and the hybrid can differ, since both use the same
  retrieval gate. Only 1 of 16 training examples (`train-style-02`) teaches an
  "insufficient evidence" reply, so the adapter may not improve abstention much.

## Limitations

- 14 calibration questions is a small sample; one additional hard negative
  could change the picture.
- The gate uses only the top-1 score. A cross-encoder reranker or a
  relevance check on the generated answer is a candidate next experiment for
  near-topic questions.

## Reproduce

```bash
uv run python -m src.calibration
```
