"""Build all model artifacts for a given embedding model.

Usage:
    python -m src.build_models                        # default: legal-bert
    python -m src.build_models --model all-minilm
"""

import argparse
import os
import pickle

import faiss
import torch

from src.config import DEFAULT_MODEL, get_model_spec, index_dir, model_dir
from src.data_loader import CATEGORIES, CATEGORY_NAMES, load_dataset
from src.embed import embed_corpus, load_encoder
from src.faiss_utils import build_index


def main():
    parser = argparse.ArgumentParser(description="Build model artifacts")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    spec = get_model_spec(args.model)
    mdir = model_dir(spec)
    idir = index_dir(spec)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(mdir, exist_ok=True)
    os.makedirs(idir, exist_ok=True)

    print(f"Model: {spec.key} ({spec.name})")

    print("Loading and saving encoder + tokenizer ...")
    model, tokenizer = load_encoder(spec, device)
    model.save_pretrained(mdir)
    tokenizer.save_pretrained(mdir)
    print(f"  embedding dim: {model.config.hidden_size}")

    print("Loading dataset ...")
    dataset = load_dataset(args.data_dir)
    sentences = dataset["sentences"]

    print(f"Embedding {len(sentences)} sentences ...")
    embeddings = embed_corpus(sentences, model, tokenizer, spec, device)

    print(f"Building FAISS index (dim={embeddings.shape[1]}) ...")
    idx = build_index(embeddings)
    faiss.write_index(idx, os.path.join(idir, "faiss.index"))

    metadata = {
        "sentences": sentences,
        "multi_labels": dataset["multi_labels"],
        "categories": CATEGORIES,
        "category_names": CATEGORY_NAMES,
    }
    with open(os.path.join(idir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"  -> {idir}/faiss.index ({idx.ntotal} vectors)")
    print("Done.")


if __name__ == "__main__":
    main()