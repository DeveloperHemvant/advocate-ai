"""
LoRA fine-tuning script for Legal Drafting AI.
Uses HuggingFace Transformers + PEFT on Llama 3 8B (or compatible) with legal_drafts JSONL.
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "legal_drafts.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "legal_llama_lora"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3-8B-Instruct"


def build_prompt(record: dict) -> str:
    """Format a single example as instruction + response for causal LM."""
    doc_type = record.get("document_type", "")
    facts = record.get("facts", "")
    draft = record.get("draft", "")
    instruction = f"Generate an Indian legal draft.\nDocument type: {doc_type}\nFacts: {facts}"
    return f"<|user|>\n{instruction}\n<|assistant|>\n{draft}"


def tokenize_and_cut(examples, tokenizer, max_length: int = 2048):
    """Tokenize and truncate to max_length."""
    texts = [build_prompt(ex) for ex in examples]
    out = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    out["labels"] = [ids.copy() for ids in out["input_ids"]]
    # Mask non-assistant part in labels (optional: mask user turn for loss only on draft)
    for i, labels in enumerate(out["labels"]):
        inp = out["input_ids"][i]
        # Simple approach: compute loss on full sequence; for Llama chat, assistant tokens follow <|assistant|>
        assistant_marker = tokenizer.encode("<|assistant|>", add_special_tokens=False)
        start = 0
        for j in range(len(inp) - len(assistant_marker) + 1):
            if inp[j: j + len(assistant_marker)] == assistant_marker:
                start = j + len(assistant_marker)
                break
        for k in range(start):
            labels[k] = -100  # ignore loss for user part
    return out


def main(
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT,
    base_model: str = DEFAULT_BASE_MODEL,
    max_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    use_4bit: bool = True,
):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Create legal_drafts.jsonl first.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load JSONL as list of dicts
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if len(data) < 10:
        logger.warning("Dataset has very few examples (%d). Consider adding more for better fine-tuning.", len(data))

    logger.info("Loading model with 4-bit quantization: %s", base_model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    def tokenize_fn(examples):
        return tokenize_and_cut(
            [dict(zip(examples.keys(), [v[i] for v in examples.values()])) for i in range(len(examples["document_type"]))],
            tokenizer,
            max_length,
        )

    ds = Dataset.from_list(data)

    def tokenize_batch(batch):
        records = [
            {"document_type": batch["document_type"][i], "facts": batch["facts"][i], "draft": batch["draft"][i]}
            for i in range(len(batch["document_type"]))
        ]
        return tokenize_and_cut(records, tokenizer, max_length)

    tokenized = ds.map(tokenize_batch, batched=True, batch_size=4, remove_columns=ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("LoRA adapter and tokenizer saved to %s", output_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--no-4bit", action="store_true", help="Use full precision (requires more VRAM)")
    args = p.parse_args()
    main(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        use_4bit=not args.no_4bit,
    )
