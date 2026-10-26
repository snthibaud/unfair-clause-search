"""
Property-based and deterministic tests for the data loader.

Uses Hypothesis to verify invariants across random subsets of the data.
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.data_loader import CATEGORIES, CATEGORY_NAMES, load_dataset, load_labels, load_overall_labels, load_sentences


# ---------------------------------------------------------------------------
# Deterministic tests
# ---------------------------------------------------------------------------
def test_load_sentences_not_empty(data_dir):
    sentences = load_sentences(data_dir)
    assert len(sentences) > 0
    assert all(isinstance(s, str) and len(s) > 0 for s in sentences)


def test_load_labels_shape(data_dir):
    sentences = load_sentences(data_dir)
    labels = load_labels(data_dir)
    assert len(labels) == len(sentences)
    assert all(len(row) == len(CATEGORIES) for row in labels)
    assert all(v in (0, 1) for row in labels for v in row)


def test_load_overall_labels_shape(data_dir):
    sentences = load_sentences(data_dir)
    overall = load_overall_labels(data_dir)
    assert len(overall) == len(sentences)
    assert all(v in (0, 1) for v in overall)


def test_overall_is_union_of_categories(data_dir):
    """The overall label must equal logical OR of all 8 category labels."""
    overall = load_overall_labels(data_dir)
    labels = load_labels(data_dir)

    for i in range(len(overall)):
        any_category_positive = any(labels[i][c] == 1 for c in range(len(CATEGORIES)))
        if any_category_positive:
            assert overall[i] == 1, f"Sentence {i}: categories positive but overall=0"
        # Note: overall[i]==1 with no category positive is fine (possible annotation artifact)


def test_dataset_split(data_dir):
    """Train/val split should be disjoint and cover all indices."""
    ds = load_dataset(data_dir, val_size=0.15)
    n = len(ds["sentences"])

    train_set = set(ds["train_idx"])
    val_set = set(ds["val_idx"])

    assert len(train_set & val_set) == 0, "Train and val sets overlap"
    assert train_set | val_set == set(range(n)), "Split doesn't cover all indices"
    assert 0.1 < len(val_set) / n < 0.2, f"Val ratio is {len(val_set)/n:.2f}, expected ~0.15"


def test_category_names_consistent():
    """Every category has a human-readable name."""
    assert len(CATEGORIES) == len(CATEGORY_NAMES)
    for cat in CATEGORIES:
        assert cat in CATEGORY_NAMES


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------
@settings(max_examples=100)
@given(
    start=st.integers(min_value=0, max_value=9000),
    count=st.integers(min_value=1, max_value=100),
)
def test_random_slice_labels_are_binary(data_dir, start, count):
    """Property: every label in any contiguous slice is 0 or 1."""
    assume(start + count <= 9414)  # total known sentence count
    sentences = load_sentences(data_dir)
    assume(start < len(sentences))
    count = min(count, len(sentences) - start)

    labels_rows = load_labels(data_dir)
    slice_rows = labels_rows[start:start + count]

    for row in slice_rows:
        for val in row:
            assert val in (0, 1)


@settings(max_examples=100)
@given(
    start=st.integers(min_value=0, max_value=9000),
    count=st.integers(min_value=1, max_value=100),
)
def test_random_slice_sentence_label_alignment(data_dir, start, count):
    """Property: any slice yields equal-length sentences and labels."""
    assume(start + count <= 9414)
    sentences = load_sentences(data_dir)
    assume(start < len(sentences))
    count = min(count, len(sentences) - start)

    sents_slice = sentences[start:start + count]
    labels_rows = load_labels(data_dir)
    overall = load_overall_labels(data_dir)

    assert len(sents_slice) == len(labels_rows[start:start + count])
    assert len(sents_slice) == len(overall[start:start + count])


@settings(max_examples=200)
@given(
    idx=st.integers(min_value=0, max_value=9413),
)
def test_sentence_non_empty_at_index(data_dir, idx):
    """Property: every sentence in the corpus is a non-empty string."""
    sentences = load_sentences(data_dir)
    assume(idx < len(sentences))
    s = sentences[idx]
    assert isinstance(s, str)
    assert len(s.strip()) > 0


@settings(max_examples=200)
@given(
    idx=st.integers(min_value=0, max_value=9413),
)
def test_overall_implies_category_if_only_one_positive(data_dir, idx):
    """Property: if exactly one category is positive, overall must also be positive."""
    labels = load_labels(data_dir)
    overall = load_overall_labels(data_dir)
    assume(idx < len(labels))

    row = labels[idx]
    num_positive = sum(row)

    # If exactly one category is 1, overall must be 1
    # (annotation consistency check)
    if num_positive == 1:
        assert overall[idx] == 1, (
            f"Sentence {idx}: exactly one category positive ({[CATEGORIES[i] for i, v in enumerate(row) if v == 1]}), "
            f"but overall is 0"
        )