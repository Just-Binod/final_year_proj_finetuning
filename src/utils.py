"""
utils.py — Seed setting, logging, HuggingFace login helper
"""

import os
import random
import numpy as np
import torch
from huggingface_hub import login


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Seed] Set to {seed}")


def hf_login(token: str = None):
    """Login to HuggingFace. Pass token or set HF_TOKEN env variable."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not set. Pass token= or set env variable HF_TOKEN.")
    login(token=token)
    print("[HF] Logged in successfully.")


def get_device():
    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[Device] CPU only — training will be very slow!")
    return "cpu"


def print_gpu_memory():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU Memory] {used:.2f}GB used / {total:.2f}GB total")
