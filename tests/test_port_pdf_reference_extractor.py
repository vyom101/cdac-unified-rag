"""
Unit tests for src/reference_extractor.py.

These validate figure/table/equation reference extraction, page-accurate index
building, and query reference detection - the metadata that powers reference-aware
retrieval (e.g. "what does Fig 1.5 show?" resolving to the correct page).
"""

from rag.extract.reference_extractor import (
    build_page_reference_index,
    extract_query_references,
    extract_references,
    is_visual_query,
    query_has_references,
    resolve_reference_pages,
)


def test_extract_figure_references_basic():
    text = "Fig 1.4 Python Code Accuracy\n\nFig 1.5 Python Code Decision tree"
    refs = extract_references(text)
    assert refs["figure_refs"] == ["1.4", "1.5"]
    assert "Fig 1.5 Python Code Decision tree" in refs["figure_captions"]


def test_extract_figure_variants():
    text = "See Figure 2.3 and fig. 4 and Fig5 for details."
    refs = extract_references(text)
    assert refs["figure_refs"] == ["2.3", "4", "5"]


def test_extract_figure_number_without_space_before_word():
    # "Fig 2.1Weka Output" - number must stop before the trailing word.
    refs = extract_references("Fig 2.1Weka Output")
    assert refs["figure_refs"] == ["2.1"]


def test_extract_table_and_equation_references():
    text = "As shown in Table 1 and Table 2.3, and Eq. (4) proves it."
    refs = extract_references(text)
    assert refs["table_refs"] == ["1", "2.3"]
    assert refs["equation_refs"] == ["4"]


def test_no_references_returns_empty_lists():
    refs = extract_references("This paragraph mentions no figures or tables at all.")
    assert refs["figure_refs"] == []
    assert refs["table_refs"] == []
    assert refs["equation_refs"] == []
    assert refs["figure_captions"] == []


def test_deduplication_preserves_order():
    text = "Fig 1.5 first, then again Fig 1.5, and Fig 1.1."
    refs = extract_references(text)
    assert refs["figure_refs"] == ["1.5", "1.1"]


def test_extract_query_references():
    q = "What is the root node condition shown in the CART Decision Tree in Fig 1.5?"
    refs = extract_query_references(q)
    assert refs["figure_refs"] == ["1.5"]
    assert refs["table_refs"] == []


def test_query_has_references():
    assert query_has_references("Explain Fig 1.5") is True
    assert query_has_references("Explain the CART algorithm") is False


def test_build_page_reference_index_is_page_accurate():
    pages = [
        (2, "For binary classification the root node splits the data."),
        (7, "Fig 1.4 Python Code Accuracy\n\nFig 1.5 Python Code Decision tree"),
        (8, "Fig 2.1 Weka Output"),
    ]
    index = build_page_reference_index(pages)
    assert index["figures"]["1.5"] == [7]
    assert index["figures"]["1.4"] == [7]
    assert index["figures"]["2.1"] == [8]
    assert "1.5" not in index["figures"].get("2", [])


def test_build_page_reference_index_multiple_pages_for_same_ref():
    pages = [
        (3, "Table 1 summarizes the dataset."),
        (9, "Table 1 (contd) continues the summary."),
    ]
    index = build_page_reference_index(pages)
    assert index["tables"]["1"] == [3, 9]


def test_resolve_reference_pages_subset():
    index = {"1.5": [7], "1.4": [7], "2.1": [8]}
    resolved = resolve_reference_pages(["1.5", "9.9"], index)
    assert resolved == {"1.5": [7]}


def test_is_visual_query_detects_figure_reference():
    assert is_visual_query("What is the root node in Fig 1.5?") is True


def test_is_visual_query_detects_visual_keywords():
    assert is_visual_query("Explain the decision tree diagram") is True
    assert is_visual_query("What does the plot show?") is True
    assert is_visual_query("Describe the confusion matrix") is True


def test_is_visual_query_false_for_conceptual_question():
    assert is_visual_query("What is entropy in machine learning?") is False
    assert is_visual_query("Define the Gini index") is False

