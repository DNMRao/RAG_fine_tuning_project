# Generation Demo: Baseline RAG vs Hybrid RAG + LoRA

Evidence for Milestone 3: the same question and the same retrieved evidence, run
through both systems.

## Setup

| Setting | Value |
|---|---|
| Command | `uv run python -m src.generation` |
| Retriever | One shared `Retriever` (MiniLM @ `1110a243`, Qdrant in-memory), `TOP_K=3`, `MIN_SCORE=0.45` |
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f7` |
| Adapter (hybrid only) | `artifacts/support-lora-adapter` (rank 8, alpha 16, `q_proj`/`v_proj`, 4 epochs, lr 0.002) |
| Decoding | Greedy (`do_sample=False`), `MAX_NEW_TOKENS=80` |
| Questions | Calibration questions only (`cal-01`, `cal-11`, `cal-09`); no held-out cases used |

Greedy decoding is deterministic: a second run gave identical outputs.
Latency varies with machine load. The values below are from one run and are
not a benchmark.

## Answer Pipeline

```
question → retrieval gate (top score < MIN_SCORE → abstained)
         → generate from "[doc-id] paragraph" evidence
         → model says "insufficient"?           → abstained
         → no citation, or cites unretrieved ID → rejected
         → otherwise                            → answered
```

A rejected answer is never shown to the customer; they see the fallback
"The approved material is insufficient. Escalate for manual review." The raw
model output is kept for analysis.

## Results

### cal-01: "If a refund is approved, where does the money go back to?"

Expected: answerable from `refunds` ("Approved refunds return to the original
payment method").
Retrieved (both systems): `refunds#0`, `refunds#2`, `subscriptions#1`

| System | Status | Reason | Output |
|---|---|---|---|
| Baseline | **rejected** | answer has no citation | If a refund is approved, the money goes back to the original payment method. |
| Hybrid | **answered** | citations match retrieved evidence | The purchase is not partially refunded within the 30-day window. The refund is not required for the purchase. [refunds] |

### cal-11: "What is the office Wi-Fi password?"

Expected: abstain (not covered by any policy; top score 0.340 < 0.45).

| System | Status | Reason | Output |
|---|---|---|---|
| Baseline | **abstained** | retrieval score below MIN_SCORE | — |
| Hybrid | **abstained** | retrieval score below MIN_SCORE | — |

### cal-09: "Can I extend my subscription for free because the platform was unavailable?"

Expected: abstain. This is a hard negative: the top score of 0.581 passes the
retrieval gate, but no policy covers outage compensation.
Retrieved (both systems): `subscriptions#0`, `subscriptions#1`, `subscriptions#2`

| System | Status | Reason | Output |
|---|---|---|---|
| Baseline | **rejected** | answer has no citation | Based on the context, it is not possible to extend your subscription for free because the platform was unavailable. The context states that the platform was unavailable for 30 days before the renewal date, and the pause begins on the date selected by the customer. This means that the customer can cancel their subscription for free, but they cannot extend their subscription for free because the platform was unavailable. |
| Hybrid | **answered** | citations match retrieved evidence | The purchase is not partially refunded. The 30-day window for the purchase is not available. [subscriptions] |

## Summary

| Outcome | Baseline | Hybrid |
|---|---|---|
| Answered | 0 | 2 |
| Abstained | 1 | 1 |
| Rejected | 2 | 0 |
| Correct content on cal-01 | Yes | No |
| Correct final decision (3 questions) | 2 (cal-11, cal-09 not answered) | 1 (cal-11) |

## Observations

1. **A valid citation does not mean a correct answer.** The hybrid was
   "answered" twice with valid citations, but both replies are wrong. The
   citation check confirms only that the cited source was retrieved, not that
   the sentence is supported by it.
2. **The adapter learned the format, not the skill.** Every hybrid reply has
   the trained shape (short sentences plus `[source]`), but the content mixes
   phrases from the training answers ("30-day window", "not partially
   refunded"). This is consistent with overfitting: 16 examples, learning
   rate 0.002, training loss falling to 0.30.
3. **The strict citation rule hides correct baseline content.** On cal-01 the
   baseline answered correctly but omitted the citation and was rejected.
   Content relevance must therefore be scored separately from the final
   status in Milestone 4.
4. **Neither model abstains by itself.** On cal-09, which passes the retrieval
   gate, neither model said the evidence was insufficient. The baseline
   invented a fact ("unavailable for 30 days before the renewal date"). Only
   1 of 16 training examples teaches the "insufficient evidence" reply.
5. **The retrieval gate works for off-topic questions.** cal-11 abstained in
   both systems without calling the model.

## Milestone 3 Checklist

- [x] One baseline response and one hybrid response for the same question and evidence (cal-01)
- [x] Valid citations (hybrid, cal-01) or a visible rejection (baseline, cal-01)
- [x] A supported abstention example (cal-11)
