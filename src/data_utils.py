import os
import json
import re
from pathlib import Path
from datasets import load_dataset

# ─────────────────────────────────────────────
# PROMPT TEMPLATES (ChatML Style for Best Fine-Tuning Performance)
# ─────────────────────────────────────────────

def format_chatml(instruction: str, user_input: str, response: str = None, tokenizer=None) -> str:
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
# CLEANING HELPERS (Safe & Permissive Boundary Limits)
# ─────────────────────────────────────────────

def is_valid_translation_pair(src: str, tgt: str) -> bool:
    if not src or not tgt: return False
    return len(str(src).strip()) >= 2 and len(str(tgt).strip()) >= 2

def is_valid_summarization(article: str, summary: str) -> bool:
    if not article or not summary: return False
    return len(str(article).strip()) >= 10 and len(str(summary).strip()) >= 5


# ─────────────────────────────────────────────
# ROBUST DATASET LOADERS
# ─────────────────────────────────────────────

def load_translation_data(n_train: int = 5000, n_val: int = 500, tokenizer=None):
    print("[Translation] Loading native OPUS-100 compilation split...")
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-ne", split="train")
    filtered = []
    
    for ex in dataset:
        trans = ex.get("translation", {})
        src = trans.get("en", "")
        tgt = trans.get("ne", "")
        if is_valid_translation_pair(src, tgt):
            filtered.append({"src": str(src), "tgt": str(tgt)})
        if len(filtered) >= (n_train + n_val):
            break

    train_data = filtered[:n_train]
    val_data = filtered[n_train:n_train + n_val]

    train_formatted = [{"text": format_translation(x["src"], x["tgt"], tokenizer)} for x in train_data]
    val_formatted = [{"text": format_translation(x["src"], x["tgt"], tokenizer), "src": x["src"], "tgt": x["tgt"]} for x in val_data]
    return train_formatted, val_formatted, val_formatted.copy()


def load_summarization_data(n_train: int = 3000, n_val: int = 300, tokenizer=None):
    print("[Summarization] Loading gold-standard csebuetnlp/xlsum Nepali split...")
    # Using full load since XL-Sum Nepali partition is relatively lightweight and avoids stream-key dropping issues
    dataset = load_dataset("csebuetnlp/xlsum", "nepali", split="train")
    filtered = []
    
    for ex in dataset:
        article = ex.get('text') or ex.get('main_text') or ''
        summary = ex.get('summary') or ''
        
        if is_valid_summarization(article, summary):
            filtered.append({"article": str(article).strip(), "summary": str(summary).strip()})
        if len(filtered) >= (n_train + n_val):
            break

    # Robust fallback strategy in case feature structure variants occur
    if len(filtered) == 0:
        print("[Warning] Retrying loader with secondary data strategy...")
        for ex in dataset:
            # Fallback to the first two keys dynamically present in row dictionary
            keys = list(ex.keys())
            if len(keys) >= 2:
                article, summary = ex[keys[0]], ex[keys[1]]
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
    print("[QA] Loading standard xquad Nepali QA split...")
    dataset = load_dataset("xquad", "xquad.ne", split="train")
    data = []
    
    for ex in dataset:
        # Check standard xquad structural mapping signatures
        context = ex.get("context") or ''
        question = ex.get("question") or ''
        answers_dict = ex.get("answers", {})
        
        # Pull text from native answers schema list
        answer = ""
        if answers_dict and "text" in answers_dict and len(answers_dict["text"]) > 0:
            answer = answers_dict["text"][0]
            
        if question and answer:
            data.append({"context": str(context).strip(), "question": str(question).strip(), "answer": str(answer).strip()})

    # Robust fallback structural lookup if xquad schema isn't directly matching top-level keys
    if len(data) == 0:
        print("[Warning] Evaluating direct mirror format layout patterns...")
        backup_dataset = load_dataset("dineshkarki/textbooks-qa-nepali", split="train")
        for ex in backup_dataset:
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
# SAVE / LOAD JSONL FILE HANDLING
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
    return data