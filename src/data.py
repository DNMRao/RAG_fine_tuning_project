"""Load and validate the three data layers before anything is indexed or trained."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

KB_DIR = Path("data/knowledge_base")
TRAIN_PATH = Path("data/training/instruction_examples.jsonl")
EVAL_PATH = Path("data/evaluation/cases.json")


class TrainingExample(BaseModel):
    id: str = Field(min_length=1)            # min_length=1 -> empty strings are rejected
    reference_ids: list[str] = Field(min_length=1)  # must cite at least one source
    context: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class EvalCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_sources: list[str]
    required_terms: list[str]
    forbidden_terms: list[str]
    should_abstain: bool


def knowledge_base_ids(kb_dir: Path = KB_DIR) -> set[str]:
    """Source IDs are the policy filenames without '.md'."""
    return {p.stem for p in kb_dir.glob("*.md")}  ## It gives the stem of the files in a set {a,b,c}.. no duplicates


def load_training(path: Path = TRAIN_PATH) -> list[TrainingExample]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [TrainingExample.model_validate_json(line) for line in lines if line.strip()]


def load_cases(path: Path = EVAL_PATH) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(item) for item in raw]


def find_problems() -> list[str]:
    """Cross-file checks Pydantic can't do alone. Empty list = data is clean."""
    kb_ids = knowledge_base_ids()
    train = load_training()
    cases = load_cases()
    problems: list[str] = []

    # Check 1: every training citation points to a real policy file
    for ex in train:
        for ref in ex.reference_ids:
            if ref not in kb_ids:
                problems.append(f"{ex.id}: unknown source '{ref}'")
    

    for eval_case in cases:
        for ref in eval_case.expected_sources:
            if ref not in kb_ids:
                problems.append(f"{eval_case.id}: unknown source '{ref}'")

    ids = [c.id for c in cases] 
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"duplicate evaluation IDs: {duplicates}") 
   
    train_questions = {ex.question.strip().casefold() for ex in train}
    for case in cases:
        if case.question.strip().casefold() in train_questions:
            problems.append(f"{case.id}: question also appears in training data")
 
    for eval_case in cases: 
      if eval_case.should_abstain != (not eval_case.expected_sources):
           problems.append(f"{eval_case.id}: should_abstain disagrees with expected_sources")

    return problems


if __name__ == "__main__":
    print(f"{len(knowledge_base_ids())} policies, {len(load_training())} training examples, "
          f"{len(load_cases())} evaluation cases")
    problems = find_problems()
    print("\n".join(problems) if problems else "All checks passed.")

