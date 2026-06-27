import os
import json
import re
from pathlib import Path
from datasets import load_dataset

# ─────────────────────────────────────────────
# PROMPT TEMPLATES (ChatML Style for Best Fine-Tuning Performance)
# CRITICAL: These match structural boundaries for Qwen and Llama-3
# ─────────────────────────────────────────────

def format_chatml(instruction: str, user_input: str, response: str = None, tokenizer=None) -> str:
    """Standardizes prompt layouts into a secure conversational framework."""
    bos = tokenizer.bos_token if (tokenizer and hasattr(tokenizer, 'bos_token')) else ""
    eos = tokenizer.eos_token if (tokenizer and hasattr(tokenizer, 'eos_token')) else "<|im_end|>"
    
    prompt = (
        f"{bos}<|im_start|>system\nतपाईं एक उपयोगी नेपाली AI सहायक हुनुहुन्छ। (You are a helpful Nepali AI assistant.)<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n{user_input.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if response is not None:
        prompt += f"{response.strip()}{eos}"
    return prompt

def format_translation(src: str, tgt: str = None, tokenizer=None) -> str:
    instruction = "Translate the following English text into clear, fluent Nepali script."
    user_input = f"English: {src}"
    return format_chatml(instruction, user_input, tgt, tokenizer)

def format_summarization(article: str, summary: str = None, tokenizer=None) -> str:
    instruction = "दिएको नेपाली लेख पढ्नुहोस् र यसको मुख्य सन्देश स्पष्ट हुने गरी छोटो सारांश तयार पार्नुहोस्। (Summarize the following article briefly.)"
    return format_chatml(instruction, article, summary, tokenizer)

def format_qa(context: str, question: str, answer: str = None, tokenizer=None) -> str:
    instruction = "दिएको सन्दर्भको आधारमा प्रश्नको उत्तर दिनुहोस्। (Answer the question based on the given context.)"
    user_input = f"सन्दर्भ (Context): {context}\nप्रश्न (Question): {question}"
    return format_chatml(instruction, user_input, answer, tokenizer)


# ─────────────────────────────────────────────
# CLEANING HELPERS
# ─────────────────────────────────────────────

def is_valid_translation_pair(src: str, tgt: str, min_chars: int = 10, max_chars: int = 300) -> bool:
    if not src or not tgt:
        return False
    if len(src) < min_chars or len(tgt) < min_chars:
        return False
    if len(src) > max_chars or len(tgt) > max_chars:
        return False
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if not devanagari_pattern.search(tgt):
        return False
    return True

def is_valid_summarization(article: str, summary: str, min_article: int = 100, max_article: int = 1500, min_summary: int = 20) -> bool:
    if not article or not summary:
        return False
    if len(article) < min_article or len(article) > max_article:
        return False
    if len(summary) < min_summary:
        return False
    return True


# ─────────────────────────────────────────────
# NEW UPDATED DATASET LOADERS (Respects GPU and API constraints)
# ─────────────────────────────────────────────

def load_translation_data(n_train: int = 5000, n_val: int = 500, tokenizer=None):
    """
    Loads up-to-date Parquet translation rows natively.
    Uses streaming to process data safely within low memory limits.
    """
    print("=" * 50)
    print("[Translation] Loading modern Parquet translation dataset...")
    print("=" * 50)
    
    dataset = load_dataset("ashokpoudel/nepali-english-translation-dataset", split="train", streaming=True)
    
    filtered = []
    for ex in dataset:
        src = ex.get('english', '')
        tgt = ex.get('nepali', '')
        if is_valid_translation_pair(src, tgt):
            filtered.append({"src": src, "tgt": tgt})
        if len(filtered) >= (n_train + n_val):
            break

    train_data = filtered[:n_train]
    val_data = filtered[n_train:n_train + n_val]

    train_formatted = [{"text": format_translation(x["src"], x["tgt"], tokenizer)} for x in train_data]
    val_formatted = [{"text": format_translation(x["src"], x["tgt"], tokenizer), "src": x["src"], "tgt": x["tgt"]} for x in val_data]
    test_formatted = val_formatted.copy()  # Clean evaluation fallback matching your pipeline logic

    print(f"[Translation] Final Dataset Split -> Train: {len(train_formatted)}, Val: {len(val_formatted)}")
    return train_formatted, val_formatted, test_formatted


def load_summarization_data(n_train: int = 3000, n_val: int = 300, tokenizer=None):
    """Loads clean, non-deprecated text summarization datasets in native Devanagari script."""
    print("=" * 50)
    print("[Summarization] Loading clean native Nepali summarization pieces...")
    print("=" * 50)
    
    dataset = load_dataset("realsanjeev/nepali-summarization-dataset", split="train", streaming=True)
    
    filtered = []
    for ex in dataset:
        article = ex.get('text', '')
        summary = ex.get('summary', '')
        if is_valid_summarization(article, summary):
            filtered.append({"article": article, "summary": summary})
        if len(filtered) >= (n_train + n_val):
            break

    train_data = filtered[:n_train]
    val_data = filtered[n_train:n_train + n_val]

    train_formatted = [{"text": format_summarization(x["article"], x["summary"], tokenizer)} for x in train_data]
    val_formatted = [{"text": format_summarization(x["article"], x["summary"], tokenizer), "article": x["article"], "summary": x["summary"]} for x in val_data]

    print(f"[Summarization] Final Dataset Split -> Train: {len(train_formatted)}, Val: {len(val_formatted)}")
    return train_formatted, val_formatted


def load_qa_data(tokenizer=None):
    """
    Loads advanced citation-grounded QA Rows.
    Falls back gracefully if gated terms aren't signed off on Hugging Face.
    """
    print("=" * 50)
    print("[QA] Loading advanced citation-grounded Nepal QA benchmark...")
    print("=" * 50)
    
    try:
        # High quality legal contextual questions and answers
        dataset = load_dataset("chhatramani/nepal-legal-qa-benchmark_v1", split="train")
    except Exception as e:
        print(f"[Warning] Gated access or script issue ({e}). Falling back to mirrored textbook QA dataset...")
        dataset = load_dataset("dineshkarki/textbooks-qa-nepali", split="train")

    data = []
    for ex in dataset:
        instruction = ex.get("instruction", "दिएको प्रश्नको सही उत्तर दिनुहोस्।")
        context = ex.get("input", "")
        answer = ex.get("output", "")
        if answer:
            data.append({"context": context, "question": instruction, "answer": answer})

    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_formatted = [{"text": format_qa(x["context"], x["question"], x["answer"], tokenizer)} for x in train_data]
    val_formatted = [{"text": format_qa(x["context"], x["question"], x["answer"], tokenizer), "context": x["context"], "question": x["question"], "answer": x["answer"]} for x in val_data]

    print(f"[QA] Final Dataset Split -> Train: {len(train_formatted)}, Val: {len(val_formatted)}")
    return train_formatted, val_formatted


# ─────────────────────────────────────────────
# SAVE / LOAD JSONL
# ─────────────────────────────────────────────

def save_jsonl(data: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[Saved] {len(data)} examples → {path}")


def load_jsonl(path: str) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    print(f"[Loaded] {len(data)} examples ← {path}")
    return data