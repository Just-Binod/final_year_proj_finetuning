"""
data_utils.py — Dataset loaders, cleaners, prompt formatters for all 3 tasks
"""

from datasets import load_dataset
import json
import re
from pathlib import Path


# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# CRITICAL: These must be IDENTICAL at train and inference time
# ─────────────────────────────────────────────

def format_translation(src: str, tgt: str = None, tokenizer=None) -> str:
    prompt = f"""### Task: Translate the following English text to Nepali.

### English:
{src.strip()}

### Nepali:
"""
    if tgt is not None:
        eos = tokenizer.eos_token if tokenizer else "</s>"
        prompt += tgt.strip() + eos
    return prompt


def format_summarization(article: str, summary: str = None, tokenizer=None) -> str:
    prompt = f"""### Task: Summarize the following Nepali news article in Nepali.

### Article:
{article.strip()}

### Summary:
"""
    if summary is not None:
        eos = tokenizer.eos_token if tokenizer else "</s>"
        prompt += summary.strip() + eos
    return prompt


def format_qa(context: str, question: str, answer: str = None, tokenizer=None) -> str:
    prompt = f"""### Task: Answer the question based on the given context in Nepali.

### Context:
{context.strip()}

### Question:
{question.strip()}

### Answer:
"""
    if answer is not None:
        eos = tokenizer.eos_token if tokenizer else "</s>"
        prompt += answer.strip() + eos
    return prompt


# ─────────────────────────────────────────────
# CLEANING HELPERS
# ─────────────────────────────────────────────

def is_valid_translation_pair(src: str, tgt: str,
                               min_chars: int = 10,
                               max_chars: int = 300) -> bool:
    """Filter out too-short, too-long, or suspicious pairs."""
    if not src or not tgt:
        return False
    if len(src) < min_chars or len(tgt) < min_chars:
        return False
    if len(src) > max_chars or len(tgt) > max_chars:
        return False
    # Skip if Nepali side has no Devanagari
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if not devanagari_pattern.search(tgt):
        return False
    return True


def is_valid_summarization(article: str, summary: str,
                            min_article: int = 100,
                            max_article: int = 1500,
                            min_summary: int = 20) -> bool:
    if not article or not summary:
        return False
    if len(article) < min_article or len(article) > max_article:
        return False
    if len(summary) < min_summary:
        return False
    return True


# ─────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────

def load_translation_data(n_train: int = 5000, n_val: int = 500, tokenizer=None,
                           min_laser_score: float = 1.05):
    """
    Load eng_Latn-npi_Deva from allenai/nllb (mined, but score-filterable —
    far higher quality than raw opus-100 once filtered).
    Falls back to opus-100 en-ne if the NLLB config/load fails.
    Augments train set with FLORES-200 'dev' split (human-quality, ~997 pairs).
    Test set remains FLORES-200 'devtest' — never trained on, so it's a clean benchmark.
    """
    filtered = []
    try:
        print(f"[Translation] Loading allenai/nllb eng_Latn-npi_Deva (laser_score >= {min_laser_score}) ...")
        raw = load_dataset("allenai/nllb", "eng_Latn-npi_Deva", split="train", streaming=True)
        for ex in raw:
            if ex.get("laser_score", 0) < min_laser_score:
                continue
            src = ex["translation"]["eng_Latn"]
            tgt = ex["translation"]["npi_Deva"]
            if is_valid_translation_pair(src, tgt):
                filtered.append({"src": src, "tgt": tgt})
            if len(filtered) >= (n_train + n_val):
                break
        print(f"[Translation] NLLB high-quality pairs collected: {len(filtered)}")
    except Exception as e:
        print(f"[Warning] allenai/nllb load failed ({e}). Falling back to opus-100 en-ne ...")
        raw = load_dataset("Helsinki-NLP/opus-100", "en-ne", split="train")
        for ex in raw:
            src = ex["translation"]["en"]
            tgt = ex["translation"]["ne"]
            if is_valid_translation_pair(src, tgt):
                filtered.append({"src": src, "tgt": tgt})
            if len(filtered) >= (n_train + n_val):
                break

    print(f"[Translation] After filtering: {len(filtered)} pairs (need {n_train + n_val})")

    train_data = filtered[:n_train]
    val_data = filtered[n_train:n_train + n_val]

    # Boost train set with FLORES-200 'dev' split (human-translated, not the devtest used for testing)
    try:
        flores_dev = load_dataset("facebook/flores", "npi_Deva-eng_Latn", split="dev")
        boost = [{"src": ex["sentence_eng_Latn"], "tgt": ex["sentence_npi_Deva"]}
                  for ex in flores_dev if ex.get("sentence_eng_Latn") and ex.get("sentence_npi_Deva")]
        train_data = train_data + boost
        print(f"[Translation] Added {len(boost)} human-quality FLORES-200 dev pairs to train set")
    except Exception as e:
        print(f"[Warning] FLORES-200 dev boost failed: {e}")

    # Format prompts
    train_formatted = [{"text": format_translation(x["src"], x["tgt"], tokenizer)} for x in train_data]
    val_formatted = [{"text": format_translation(x["src"], x["tgt"], tokenizer),
                      "src": x["src"], "tgt": x["tgt"]} for x in val_data]

    # Load flores200 test set (devtest — held out, never trained on)
    print("[Translation] Loading flores200 test set ...")
    try:
        flores = load_dataset("facebook/flores", "npi_Deva-eng_Latn", split="devtest")
        test_formatted = []
        for ex in flores:
            src = ex.get("sentence_eng_Latn", "")
            tgt = ex.get("sentence_npi_Deva", "")
            if src and tgt:
                test_formatted.append({
                    "text": format_translation(src, tgt, tokenizer),
                    "src": src, "tgt": tgt
                })
    except Exception as e:
        print(f"[Warning] flores200 failed: {e}. Using val split as test.")
        test_formatted = val_formatted

    print(f"[Translation] Train: {len(train_formatted)}, Val: {len(val_formatted)}, Test: {len(test_formatted)}")
    return train_formatted, val_formatted, test_formatted


def load_summarization_data(n_train: int = 3000, n_val: int = 300, tokenizer=None):
    """Load xlsum nepali — BBC quality, best available."""
    print("[Summarization] Loading xlsum nepali ...")
    raw_train = load_dataset("csebuetnlp/xlsum", "nepali", split="train")
    raw_val = load_dataset("csebuetnlp/xlsum", "nepali", split="validation")

    def process_split(raw, n):
        results = []
        for ex in raw:
            article = ex.get("text", "")
            summary = ex.get("summary", "")
            if is_valid_summarization(article, summary):
                results.append({"article": article, "summary": summary})
            if len(results) >= n:
                break
        return results

    train_data = process_split(raw_train, n_train)
    val_data = process_split(raw_val, n_val)

    train_formatted = [{"text": format_summarization(x["article"], x["summary"], tokenizer)}
                       for x in train_data]
    val_formatted = [{"text": format_summarization(x["article"], x["summary"], tokenizer),
                      "article": x["article"], "summary": x["summary"]}
                     for x in val_data]

    print(f"[Summarization] Train: {len(train_formatted)}, Val: {len(val_formatted)}")
    return train_formatted, val_formatted


def load_qa_data(tokenizer=None):
    """
    Load xquad nepali — only gold-standard Nepali QA.
    80/20 split since only one split available.
    """
    print("[QA] Loading xquad.ne ...")
    try:
        raw = load_dataset("xquad", "xquad.ne", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"[Warning] 'xquad' loading script failed ({e}). Trying mirror 'google/xquad' ...")
        raw = load_dataset("google/xquad", "xquad.ne", split="validation")

    data = []
    for ex in raw:
        context = ex.get("context", "")
        question = ex.get("question", "")
        answers = ex.get("answers", {})
        answer_texts = answers.get("text", [])
        if context and question and answer_texts:
            answer = answer_texts[0]  # take first answer
            data.append({"context": context, "question": question, "answer": answer})

    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_formatted = [{"text": format_qa(x["context"], x["question"], x["answer"], tokenizer)}
                       for x in train_data]
    val_formatted = [{"text": format_qa(x["context"], x["question"], x["answer"], tokenizer),
                      "context": x["context"], "question": x["question"], "answer": x["answer"]}
                     for x in val_data]

    print(f"[QA] Train: {len(train_formatted)}, Val: {len(val_formatted)}")
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
            data.append(json.loads(line.strip()))
    print(f"[Loaded] {len(data)} examples ← {path}")
    return data
