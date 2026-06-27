"""
training_utils.py — Unsloth model loading, QLoRA config, trainer builder
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch


# ─────────────────────────────────────────────
# MODEL CONFIGS
# ─────────────────────────────────────────────

MODEL_CONFIGS = {
    "qwen3-8b": {
        "model_name": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "max_seq_length": 512,
        "hf_prefix": "iwasbinod/qwen3-8b-nepali",
    },
    "llama32-3b": {
        "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "max_seq_length": 512,
        "hf_prefix": "iwasbinod/llama32-3b-nepali",
    },
}

TASK_SUFFIX = {
    "translation": "translation-qlora",
    "summarization": "summarization-qlora",
    "qa": "qa-qlora",
}


def load_model(model_key: str):
    """Load 4-bit quantized model with Unsloth."""
    cfg = MODEL_CONFIGS[model_key]
    print(f"[Model] Loading {cfg['model_name']} ...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,           # auto-detect
        load_in_4bit=True,
    )
    print(f"[Model] Loaded successfully.")
    return model, tokenizer


def apply_qlora(model, r: int = 16, lora_alpha: int = 32):
    """Apply QLoRA adapter to model."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves VRAM on T4
        random_state=42,
        use_rslora=False,
    )
    print(f"[QLoRA] Applied. r={r}, lora_alpha={lora_alpha}")
    return model


def build_trainer(model, tokenizer, train_data: list, output_dir: str,
                  num_epochs: int = 3, batch_size: int = 2,
                  grad_accum: int = 4, lr: float = 2e-4,
                  max_seq_length: int = 512):
    """Build SFTTrainer ready to run."""

    train_dataset = Dataset.from_list(train_data)

    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,  # effective batch = 8
        learning_rate=lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        seed=42,
        report_to="none",  # disable wandb
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,   # trl >=0.12 renamed `tokenizer` -> `processing_class`
        train_dataset=train_dataset,
        args=sft_config,
    )
    return trainer


def save_and_push(model, tokenizer, model_key: str, task: str,
                  local_dir: str = None, push: bool = True):
    """Save adapter locally and push to HuggingFace."""
    repo_id = f"{MODEL_CONFIGS[model_key]['hf_prefix']}-{TASK_SUFFIX[task]}"
    local_dir = local_dir or f"outputs/adapters/{repo_id.split('/')[-1]}"

    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    print(f"[Save] Adapter saved to {local_dir}")

    if push:
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"[HF] Pushed to https://huggingface.co/{repo_id}")

    return repo_id
