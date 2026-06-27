# Fine-tuning and Benchmarking Small Open-Source LLMs for Low-Resource Nepali NLP

**BE Computer Engineering Final Year Project**
**Author:** Binod (iwasbinod) | NAST, Dhangadhi | Pokhara University
**Supervisor:** Mr. Sunil Bahadur Bist

---

## Overview

This project fine-tunes and benchmarks small open-source LLMs for three low-resource Nepali NLP tasks:
- **Translation** (English → Nepali)
- **Summarization** (Nepali news articles)
- **Question Answering** (Nepali)

### Models
| Model | Parameters | Role |
|---|---|---|
| Qwen3-8B | 8B | High-performance model |
| Llama-3.2-3B | 3B | Lightweight / efficient model |

### Method
- QLoRA fine-tuning via [Unsloth](https://github.com/unslothai/unsloth)
- 4-bit quantization (runs on Kaggle free T4 GPU)
- PEFT adapter-only training

---

## Datasets

| Task | Dataset | Size Used |
|---|---|---|
| Translation | Helsinki-NLP/opus-100 (ne-en) | 5,000 train pairs |
| Summarization | csebuetnlp/xlsum (nepali) | 3,000 articles |
| QA | xquad (xquad.ne) | ~1,190 examples |
| Translation eval | facebook/flores (npi-eng) | Full devtest |

---

## HuggingFace Adapters

All adapters at: [https://huggingface.co/iwasbinod](https://huggingface.co/iwasbinod)

```
iwasbinod/qwen3-8b-nepali-translation-qlora
iwasbinod/qwen3-8b-nepali-summarization-qlora
iwasbinod/qwen3-8b-nepali-qa-qlora
iwasbinod/llama32-3b-nepali-translation-qlora
iwasbinod/llama32-3b-nepali-summarization-qlora
iwasbinod/llama32-3b-nepali-qa-qlora
```

---

## Repository Structure

```
final_year_proj_finetuning/
├── notebooks/          # Run in order: 00 → 01 → 02–07 → 08 → 09
├── src/                # Reusable Python modules
├── configs/            # Per-model, per-task YAML configs
├── data/samples/       # 50-example samples (committed)
├── results/            # Evaluation tables and plots
└── docs/               # Dataset links and references
```

## Run Order

1. `00_base_evaluation.ipynb` — Benchmark base models (run FIRST)
2. `01_data_preprocessing.ipynb` — Load and format all datasets
3. `02–07` training notebooks — One per model × task
4. `08_evaluation_comparison.ipynb` — Generate comparison table
5. `09_push_to_hf.ipynb` — Push adapters to HuggingFace

---

## Setup

```bash
pip install -r requirements.txt
```

Add your HuggingFace token as `HF_TOKEN` in Kaggle secrets.

---

## GitHub

[https://github.com/Just-Binod/final_year_proj_finetuning](https://github.com/Just-Binod/final_year_proj_finetuning)
