"""Answer questions from retrieved evidence with the base model (baseline) or base + LoRA (hybrid)."""

import re
import time
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ADAPTER_PATH, BASE_MODEL, BASE_MODEL_REVISION, MAX_NEW_TOKENS
from src.ingestion import build_chunks, load_documents
from src.prompts import build_messages
from src.retrieval import Retriever, SearchHit

ABSTAIN_MESSAGE = "The approved material is insufficient. Escalate for manual review."
CITATION_PATTERN = re.compile(r"\[([a-z0-9-]+)\]")


@dataclass(frozen=True)
class AnswerResult:
    question: str
    status: str  # "answered", "abstained", or "rejected"
    reply: str  # what the customer would see
    raw_output: str  # what the model generated ("" if retrieval abstained)
    retrieved_ids: list[str]  # chunk IDs given to the model, best first
    top_score: float | None
    citations: list[str]  # source IDs found in the model output
    reason: str  # why this status was chosen
    latency_seconds: float


def load_model(use_adapter: bool):
    """Base model alone (baseline RAG) or base model + saved adapter (hybrid)."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
    if use_adapter:
        if not ADAPTER_PATH.exists():
            raise FileNotFoundError(f"No adapter at {ADAPTER_PATH}; run: uv run python -m src.training")
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, context: str, question: str) -> str:
    """Greedy-decode an answer to the shared prompt; return only the new text."""
    prompt = tokenizer.apply_chat_template(
        build_messages(context, question), tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        generated = model.generate(
            **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def format_context(hits: list[SearchHit]) -> str:
    """Write evidence as '[document-id] paragraph' lines: the same layout as the training contexts."""
    # chunk.text is "Title\nparagraph"; drop the title so the layout matches training.
    return "\n".join(
        f"[{hit.chunk.document_id}] {hit.chunk.text.split(chr(10), 1)[-1]}" for hit in hits
    )


def answer_question(question: str, retriever: Retriever, model, tokenizer) -> AnswerResult:
    """Retrieve, gate, generate, and validate citations for one question."""
    start = time.perf_counter()
    hits = retriever.retrieve_evidence(question)
    if not hits:
        return AnswerResult(
            question, "abstained", ABSTAIN_MESSAGE, "", [], None, [],
            "retrieval score below MIN_SCORE", time.perf_counter() - start,
        )

    raw = generate_reply(model, tokenizer, format_context(hits), question)
    citations = CITATION_PATTERN.findall(raw)
    retrieved_sources = {hit.chunk.document_id for hit in hits}
    unretrieved = sorted(set(citations) - retrieved_sources)

    if "insufficient" in raw.lower():
        status, reply, reason = "abstained", ABSTAIN_MESSAGE, "model reported insufficient evidence"
    elif not citations:
        # Strict rule: an uncited answer cannot be verified, so it never reaches the customer.
        status, reply, reason = "rejected", ABSTAIN_MESSAGE, "answer has no citation"
    elif unretrieved:
        status, reply, reason = "rejected", ABSTAIN_MESSAGE, f"cites sources not retrieved: {unretrieved}"
    else:
        status, reply, reason = "answered", raw, "citations match retrieved evidence"

    return AnswerResult(
        question, status, reply, raw, [hit.chunk.chunk_id for hit in hits],
        hits[0].score, citations, reason, time.perf_counter() - start,
    )


if __name__ == "__main__":
    retriever = Retriever(build_chunks(load_documents()))  # built once, shared by both systems
    systems = {"baseline": load_model(use_adapter=False), "hybrid": load_model(use_adapter=True)}

    for question in [
        "If a refund is approved, where does the money go back to?",  # cal-01: answerable
        "What is the office Wi-Fi password?",  # cal-11: abstains at retrieval
        "Can I extend my subscription for free because the platform was unavailable?",  # cal-09: passes the gate
    ]:
        print(f"Q: {question}")
        for name, (model, tokenizer) in systems.items():
            result = answer_question(question, retriever, model, tokenizer)
            print(f"  [{name:8}] {result.status:9} | {result.reason} | {result.latency_seconds:.1f}s")
            print(f"             retrieved: {result.retrieved_ids}")
            print(f"             output:    {result.raw_output or '-'}")
        print()

