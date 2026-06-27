"""
evaluation.py — BLEU (translation), ROUGE (summarization), F1/EM (QA)
"""

import re
import string
from collections import Counter
import sacrebleu
from rouge_score import rouge_scorer
import pandas as pd
from unsloth import FastLanguageModel
from src.data_utils import format_translation, format_summarization, format_qa


# ─────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────

def generate_output(model, tokenizer, prompt: str,
                    max_new_tokens: int = 128,
                    temperature: float = 0.1) -> str:
    """Run inference and extract only the generated part after the prompt."""
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,         # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_len:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return decoded


# ─────────────────────────────────────────────
# TRANSLATION — BLEU
# ─────────────────────────────────────────────

def evaluate_translation(model, tokenizer, val_data: list,
                          n_samples: int = 100) -> dict:
    """Compute BLEU on translation val set."""
    print(f"[Eval] Translation BLEU on {n_samples} samples ...")
    hypotheses = []
    references = []

    for ex in val_data[:n_samples]:
        prompt = format_translation(ex["src"])   # no target — inference mode
        pred = generate_output(model, tokenizer, prompt, max_new_tokens=150)
        hypotheses.append(pred)
        references.append(ex["tgt"])

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    result = {
        "bleu": round(bleu.score, 2),
        "hypotheses": hypotheses[:5],   # sample outputs for inspection
        "references": references[:5],
    }
    print(f"[Translation] BLEU = {result['bleu']}")
    return result


# ─────────────────────────────────────────────
# SUMMARIZATION — ROUGE
# ─────────────────────────────────────────────

def evaluate_summarization(model, tokenizer, val_data: list,
                            n_samples: int = 100) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L on summarization val set."""
    print(f"[Eval] Summarization ROUGE on {n_samples} samples ...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    r1_scores, r2_scores, rL_scores = [], [], []
    hypotheses, references_list = [], []

    for ex in val_data[:n_samples]:
        prompt = format_summarization(ex["article"])  # no summary — inference mode
        pred = generate_output(model, tokenizer, prompt, max_new_tokens=128)
        ref = ex["summary"]

        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rL_scores.append(scores["rougeL"].fmeasure)
        hypotheses.append(pred)
        references_list.append(ref)

    result = {
        "rouge1": round(sum(r1_scores) / len(r1_scores) * 100, 2),
        "rouge2": round(sum(r2_scores) / len(r2_scores) * 100, 2),
        "rougeL": round(sum(rL_scores) / len(rL_scores) * 100, 2),
        "hypotheses": hypotheses[:5],
        "references": references_list[:5],
    }
    print(f"[Summarization] ROUGE-1={result['rouge1']}, ROUGE-2={result['rouge2']}, ROUGE-L={result['rougeL']}")
    return result


# ─────────────────────────────────────────────
# QA — Exact Match + F1
# ─────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    ref_tokens = normalize_answer(ref).split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_qa(model, tokenizer, val_data: list,
                n_samples: int = 100) -> dict:
    """Compute Exact Match and Token F1 on QA val set."""
    print(f"[Eval] QA EM+F1 on {n_samples} samples ...")
    em_scores, f1_scores = [], []
    hypotheses, references_list = [], []

    for ex in val_data[:n_samples]:
        prompt = format_qa(ex["context"], ex["question"])  # no answer — inference
        pred = generate_output(model, tokenizer, prompt, max_new_tokens=64)
        ref = ex["answer"]

        em = float(normalize_answer(pred) == normalize_answer(ref))
        f1 = token_f1(pred, ref)
        em_scores.append(em)
        f1_scores.append(f1)
        hypotheses.append(pred)
        references_list.append(ref)

    result = {
        "exact_match": round(sum(em_scores) / len(em_scores) * 100, 2),
        "f1": round(sum(f1_scores) / len(f1_scores) * 100, 2),
        "hypotheses": hypotheses[:5],
        "references": references_list[:5],
    }
    print(f"[QA] EM={result['exact_match']}, F1={result['f1']}")
    return result


# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────

def build_results_table(results: dict) -> pd.DataFrame:
    """
    results = {
        "Qwen3-8B Base": {"bleu": 3.2, "rouge1": 8.1, "rougeL": 7.2, "f1": 12.3},
        "Qwen3-8B Fine-tuned": {"bleu": 28.4, ...},
        ...
    }
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df
