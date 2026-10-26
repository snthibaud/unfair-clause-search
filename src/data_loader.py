"""
Data loader for the CLAUDETTE ToS dataset.

Loads aligned sentences and multi-label annotations from the data/ directory.
"""

from pathlib import Path
from typing import Optional

from sklearn.model_selection import train_test_split


CATEGORIES = ["A", "CH", "CR", "J", "LAW", "LTD", "TER", "USE"]

CATEGORY_NAMES = {
    "A": "Arbitration",
    "CH": "Unilateral Change",
    "CR": "Content Removal",
    "J": "Jurisdiction",
    "LAW": "Choice of Law",
    "LTD": "Limitation of Liability",
    "TER": "Termination",
    "USE": "Unilateral Change (Usage)",
}


def load_sentences(data_dir: str | Path = "data") -> list[str]:
    """Load all sentences from the Sentences directory, sorted by filename."""
    sentences_dir = Path(data_dir) / "Sentences"
    sentences: list[str] = []
    for path in sorted(sentences_dir.glob("*.txt")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
    return sentences


def _load_category_labels(labels_dir: Path, cat: str) -> list[int]:
    """Load binary labels for one unfairness category."""
    cat_labels: list[int] = []
    cat_dir = labels_dir / f"Labels_{cat}"
    for path in sorted(cat_dir.glob("*.txt")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    cat_labels.append(1 if line == "1" else 0)
    return cat_labels


def load_labels(data_dir: str | Path = "data", categories: Optional[list[str]] = None) -> list[list[int]]:
    """
    Load multi-label annotations.

    Args:
        data_dir: Path to the data directory.
        categories: List of category keys to load (defaults to all 8).

    Returns:
        List of lists, each inner list is [label_A, label_CH, ..., label_USE]
        where each label is 0 (fair) or 1 (unfair).
    """
    if categories is None:
        categories = CATEGORIES

    labels_dir = Path(data_dir)
    labels_per_category = [_load_category_labels(labels_dir, cat) for cat in categories]

    # Transpose: from (num_categories, num_sentences) → (num_sentences, num_categories)
    num_sentences = len(labels_per_category[0])
    return [[labels_per_category[c][i] for c in range(len(categories))] for i in range(num_sentences)]


def load_overall_labels(data_dir: str | Path = "data") -> list[int]:
    """Load the overall (union) unfairness labels."""
    labels_dir = Path(data_dir) / "Labels"
    overall: list[int] = []
    for path in sorted(labels_dir.glob("*.txt")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    overall.append(1 if line == "1" else 0)
    return overall


def load_dataset(data_dir: str | Path = "data", val_size: float = 0.15, random_state: int = 42):
    """
    Load the full dataset: sentences, multi-label vectors, overall labels.
    Returns train/val splits.
    """
    sentences = load_sentences(data_dir)
    multi_labels = load_labels(data_dir)
    overall = load_overall_labels(data_dir)

    assert len(sentences) == len(multi_labels) == len(overall), (
        f"Mismatch: {len(sentences)} sentences, {len(multi_labels)} label sets, {len(overall)} overall"
    )

    # Stratified split using overall labels as the stratum
    indices = list(range(len(sentences)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=random_state, stratify=overall
    )

    return {
        "sentences": sentences,
        "multi_labels": multi_labels,
        "overall": overall,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }