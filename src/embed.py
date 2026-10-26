"""Embedding module using HuggingFace AutoModel with mean pooling."""

from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

if (n := os.cpu_count()):
    torch.set_num_threads(n)

from src.config import ModelSpec


def load_encoder(spec: ModelSpec, device: str = "cpu"):
    """Load encoder and tokenizer from HuggingFace by ModelSpec."""
    return _load(spec.name, device)


def load_encoder_local(model_dir: str, device: str = "cpu"):
    """Load encoder and tokenizer from a local directory."""
    return _load(model_dir, device)


def _load(source: str, device: str = "cpu"):
    """Shared loading logic for both HF and local paths."""
    model = AutoModel.from_pretrained(source, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def embed_batch(
    texts: list[str],
    model,
    tokenizer,
    max_length: int,
    device: str = "cpu",
) -> np.ndarray:
    """Embed a list of texts, returning a (len(texts), dim) float32 array."""
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=mask)

    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    mask_exp = mask.unsqueeze(-1).expand(hidden.size()).float()
    pooled = torch.sum(hidden * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)
    return pooled.cpu().numpy().astype(np.float32)


@torch.inference_mode()
def embed_corpus(
    sentences: list[str],
    model,
    tokenizer,
    spec: ModelSpec,
    device: str = "cpu",
) -> np.ndarray:
    """Embed a full corpus, batching automatically."""
    embeddings: list[np.ndarray] = []
    bs = spec.batch_size
    for i in tqdm(range(0, len(sentences), bs), desc=f"Embedding ({spec.key})"):
        batch = sentences[i : i + bs]
        embeddings.append(embed_batch(batch, model, tokenizer, spec.max_length, device))
    return np.vstack(embeddings).astype(np.float32)