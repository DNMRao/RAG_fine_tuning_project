"""Train a LoRA adapter on the support examples and save it with a training summary."""

import hashlib
import json
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    ADAPTER_PATH, BASE_MODEL, BASE_MODEL_REVISION, LEARNING_RATE,
    MAX_SEQUENCE_LENGTH, RANDOM_SEED, TRAIN_EPOCHS,
)
from src.data import TRAIN_PATH, TrainingExample, find_problems, load_training
from src.prompts import SYSTEM_MESSAGE, build_messages

DEVICE = torch.device("cpu")
BATCH_SIZE = 2
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def set_seed(seed: int) -> None:
    """Fix every random number generator so reruns give the same adapter."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_prompt_and_answer(tokenizer, example: TrainingExample) -> tuple[str, str]:
    """Return (prompt the model sees, answer it must learn) using the shared prompt format."""
    prompt = tokenizer.apply_chat_template(
        build_messages(example.context, example.question),
        tokenize=False, add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        build_messages(example.context, example.question, example.answer),
        tokenize=False,
    )
    if not full_text.startswith(prompt):
        raise ValueError(f"{example.id}: training text does not start with the inference prompt")
    return prompt, full_text[len(prompt):]


class InstructionDataset(Dataset):
    """Tokenized examples whose labels cover only the answer tokens."""

    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_length: int) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        prompt, answer = self.pairs[index]
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        token_ids = self.tokenizer.encode(prompt + answer, add_special_tokens=False)
        if token_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Tokenization changed the prompt/answer boundary.")
        if len(token_ids) <= len(prompt_ids) or len(token_ids) > self.max_length:
            raise ValueError(f"Example {index}: empty answer or longer than {self.max_length} tokens.")

        padding = self.max_length - len(token_ids)
        input_ids = torch.tensor(token_ids + [self.tokenizer.pad_token_id] * padding)
        attention_mask = torch.tensor([1] * len(token_ids) + [0] * padding)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100      # don't learn to reproduce the prompt
        labels[attention_mask == 0] = -100    # don't learn from padding
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train() -> dict:
    problems = find_problems()
    if problems:
        raise ValueError("Fix the data before training:\n" + "\n".join(problems))
    set_seed(RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    examples = load_training()
    dataset = InstructionDataset(
        [split_prompt_and_answer(tokenizer, ex) for ex in examples], tokenizer, MAX_SEQUENCE_LENGTH
    )
    for index in range(len(dataset)):  # validate every example before any training starts
        dataset[index]
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False
    base_model.loss_type = "ForCausalLM"
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES, bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    model.to(DEVICE)
    model.train()
    step_losses: list[float] = []
    epoch_losses: list[float] = []
    for epoch in range(TRAIN_EPOCHS):
        current: list[float] = []
        for batch in loader:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; inspect the token labels.")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            current.append(float(loss.detach()))
        step_losses.extend(current)
        epoch_losses.append(round(sum(current) / len(current), 4))
        print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}: mean loss = {epoch_losses[-1]:.4f}")
    model.eval()

    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    model.peft_config["default"].revision = BASE_MODEL_REVISION
    model.save_pretrained(str(ADAPTER_PATH), save_embedding_layers=False)
    tokenizer.save_pretrained(str(ADAPTER_PATH))

    summary = {
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "training_data": str(TRAIN_PATH),
        "training_sha256": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "num_examples": len(examples),
        "prompt_format": "native chat template; system=SYSTEM_MESSAGE; "
                         "user='Context:\\n{context}\\n\\nQuestion: {question}'; loss on assistant turn only",
        "system_message": SYSTEM_MESSAGE,
        "lora": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": LORA_TARGET_MODULES},
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "epochs": TRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": RANDOM_SEED,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "epoch_losses": epoch_losses,
        "step_losses": [round(x, 4) for x in step_losses],
        "adapter_path": str(ADAPTER_PATH),
    }
    (ADAPTER_PATH / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = train()
    print(f"\nAdapter saved to {summary['adapter_path']}")

