import os
import json
import re
from pathlib import Path
from datasets import load_dataset

# ─────────────────────────────────────────────
# PROMPT TEMPLATES (ChatML Style for Best Fine-Tuning Performance)
# ─────────────────────────────────────────────

def format_chatml(instruction: str, user_input: str, response: str = None, tokenizer=None) -> str:
    """Standardizes prompt layouts into a secure conversational framework."""
    bos = tokenizer.bos_token if (tokenizer and hasattr(tokenizer, 'bos_token')) else ""
    eos = tokenizer.eos_token if (tokenizer and hasattr(tokenizer, 'eos_token')) else "<|im_end|>"
    
    prompt = (
        f"{bos}<|im_start|>system\nतपाईं एक उपयोगी नेपाली AI सहायक हुनुहुन्छ।<|im_end|>\n"
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
    instruction = "दिएको नेपाली लेख पढ्नुहोस् र यसको मुख्य सन्देश स्पष्ट हुने गरी छोटो सारांश तयार पार्नुहोस्।"
    return format_chatml(instruction, article, summary, tokenizer)

def format_qa(context: str, question: str, answer: str = None, tokenizer=None) -> str:
    instruction = "दिएको सन्दर्भको आधारमा प्रश्नको उत्तर दिनुहोस्।"
    user_input = f"सन्दर्भ (Context): {context}\nप्रश्न (Question): {question}"
    return format_chatml(instruction, user_input, answer, tokenizer)


# ─────────────────────────────────────────────
# CLEANING HELPERS (With Safe Boundary Relaxations)
# ─────────────────────────────────────────────

def is_valid_translation_pair(src: str, tgt: str, min_chars: int = 2, max_chars: int = 1000) -> bool:
    if not src or not tgt:
        return False
    src_str, tgt_str = str(src).strip(), str(tgt).strip()
    if len(src_str) < min_chars or len(tgt_str) < min_chars:
        return False
    if len(src_str) > max_chars or len(tgt_str) > max_chars:
        return False
    return True

def is_valid_summarization(article: str, summary: str, min_article: int = 10, max_article: int = 10000, min_summary: int = 5) -> bool:
    if not article or not summary:
        return False
    if len(str(article).strip()) < min_article or len(str(article).strip()) > max_article:
        return False
    if len(str(summary).strip()) < min_summary:
        return False
    return True


# ─────────────────────────────────────────────
# ROBUST DATASET LOADERS
# ─────────────────────────────────────────────

def load_translation_data(n_train: int = 5000, n_val: int = 500, tokenizer=None):
    print("=" * 50)
    print("[Translation] Streaming modern Parquet translation dataset...")
    print("=" * 50)
    
    dataset = load_dataset("ashokpoudel/nepali-english-translation-dataset", split="train", streaming=True)
    filtered = []
    
    for ex in dataset:
        # Check all typical naming patterns for translation pairs
        src = ex.get('english') or ex.get('en') or ex.get('src') or ''
        tgt = ex.get('nepali') or ex.get('ne') or ex.get('tgt') or ''
        
        if is_valid_translation_pair(src, tgt):
            filtered.append({"src": str(src), "tgt": str(tgt)})
        if len(filtered) >= (n_train + n_val):
            break

    # If streaming dictionary keys completely fail, use the gold-standard OPUS backup split safely
    if len(filtered) < (n_train + n_val):
        print("[Warning] High-quality stream returned insufficient rows. Utilizing default pipeline backup...")
        try:
            backup_raw = load_dataset("Helsinki-NLP/opus-100", "en-ne", split="train")
            for ex in list(backup_raw)[:(n_train + n_val + 200)]:
                src = ex["translation"]["en"]
                tgt = ex["translation"]["ne"]
                if is_valid_translation_pair(src, tgt) and {"src": str(src), "tgt": str(tgt)} not in filtered:
                    filtered.append({"src": str(src), "tgt": str(tgt)})
        except Exception as fallback_err:
            print(f"Backup layout failed: {fallback_err}")

    train_data = filtered[:n_train]
    val_data = filtered[n_train:n_train + n_val]

    train_formatted = [{"text": format_translation(x["src"], x["tgt"], tokenizer)} for x in train_data]
    val_formatted = [{"text": format_translation(x["src"], x["tgt"], tokenizer), "src": x["src"], "tgt": x["tgt"]} for x in val_data]
    test_formatted = val_formatted.copy()

    print(f"[Translation] Final Dataset Split -> Train: {len(train_formatted)}, Val: {len(val_formatted)}")
    return train_formatted, val_formatted, test_formatted


def load_summarization_data(n_train: int = 3000, n_val: int = 300, tokenizer=None):
    print("=" * 50)
    print("[Summarization] Streaming clean native Nepali summarization pieces...")
    print("=" * 50)
    
    dataset = load_dataset("realsanjeev/nepali-summarization-dataset", split="train", streaming=True)
    filtered = []
    
    for ex in dataset:
        article = ex.get('text') or ex.get('article') or ex.get('news') or ''
        summary = ex.get('summary') or ex.get('abstract') or ''
        
        if is_valid_summarization(article, summary):
            filtered.append({"article": str(article), "summary": str(summary)})
        if len(filtered) >= (n_train + n_val):
            break

    train_data = filtered[:n_train]
    val_data = filtered[n_train:n_train + n_val]

    train_formatted = [{"text": format_summarization(x["article"], x["summary"], tokenizer)} for x in train_data]
    val_formatted = [{"text": format_summarization(x["article"], x["summary"], tokenizer), "article": x["article"], "summary": x["summary"]} for x in val_data]

    print(f"[Summarization] Final Dataset Split -> Train: {len(train_formatted)}, Val: {len(val_formatted)}")
    return train_formatted, val_formatted


def load_qa_data(tokenizer=None):
    print("=" * 50)
    print("[QA] Loading advanced citation-grounded Nepal QA benchmark...")
    print("=" * 50)
    
    # Target the reliable textbook dataset directly to ensure structural compatibility
    try:
        dataset = load_dataset("dineshkarki/textbooks-qa-nepali", split="train")
    except Exception as e:
        print(f"Primary fetch hit an issue: {e}. Attempting cross-compatible backup...")
        dataset = load_dataset("chhatramani/nepal-legal-qa-benchmark_v1", split="train")

    data = []
    for ex in dataset:
        # Cross-evaluate all standard layout signatures safely
        question = ex.get("question") or ex.get("instruction") or ''
        context = ex.get("context") or ex.get("input") or 'दिएको सन्दर्भ विवरण'
        answer = ex.get("answer") or ex.get("output") or ''
        
        if question and answer:
            data.append({"context": str(context), "question": str(question), "answer": str(answer)})

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