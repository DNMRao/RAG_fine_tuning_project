"""One prompt format shared by training and generation, so the adapter is tested on what it learned."""

SYSTEM_MESSAGE = (
    "You are a Northstar support assistant. Answer only from the context. "
    "Start with the direct answer, then the next action. Cite source IDs in square brackets. "
    "If the context does not answer the question, say the approved material is "
    "insufficient and escalate for manual review."
)


def build_messages(context: str, question: str, answer: str | None = None) -> list[dict[str, str]]:
    """Chat messages for one example; include `answer` only when building training text."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages

