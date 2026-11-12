from rag.extract.reference_extractor import (
    build_page_reference_index,
    extract_query_references,
    extract_references,
    is_visual_query,
    query_has_references,
    resolve_reference_pages,
)


def test_extract_figure_references_and_captions():
    refs = extract_references(
        "Fig 1.4 Python Code Accuracy\n\nFig 1.5 Python Code Decision tree"
    )
    assert refs["figure_refs"] == ["1.4", "1.5"]
    assert "Fig 1.5 Python Code Decision tree" in refs["figure_captions"]


def test_extract_reference_variants():
    refs = extract_references(
        "See Figure 2.3, fig. 4, Fig5, Table 2.3 and Eq. (4)."
    )
    assert refs["figure_refs"] == ["2.3", "4", "5"]
    assert refs["table_refs"] == ["2.3"]
    assert refs["equation_refs"] == ["4"]


def test_number_stops_before_trailing_word():
    assert extract_references("Fig 2.1Weka Output")["figure_refs"] == ["2.1"]


def test_deduplication_preserves_order():
    refs = extract_references("Fig 1.5 first, Fig 1.5 again, then Fig 1.1.")
    assert refs["figure_refs"] == ["1.5", "1.1"]


def test_query_reference_detection():
    query = "What is the root condition shown in Fig 1.5?"
    assert extract_query_references(query)["figure_refs"] == ["1.5"]
    assert query_has_references(query) is True
    assert query_has_references("Explain the CART algorithm") is False


def test_page_reference_index_is_accurate():
    index = build_page_reference_index([
        (2, "No reference here."),
        (7, "Fig 1.4 Accuracy\nFig 1.5 Decision tree"),
        (8, "Fig 2.1 Weka Output"),
        (9, "Table 1 summary"),
        (10, "Table 1 continued"),
    ])
    assert index["figures"]["1.5"] == [7]
    assert index["figures"]["2.1"] == [8]
    assert index["tables"]["1"] == [9, 10]
    assert resolve_reference_pages(
        ["1.5", "9.9"], index["figures"]
    ) == {"1.5": [7]}


def test_visual_query_detection():
    assert is_visual_query("What is shown in Fig 1.5?") is True
    assert is_visual_query("Explain the decision tree diagram") is True
    assert is_visual_query("Describe the confusion matrix") is True
    assert is_visual_query("What is entropy in machine learning?") is False
