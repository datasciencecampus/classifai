"""Tests the EmbeddingHandler class."""

from classifai.embedding import EmbeddingHandler


def test_create_query_texts(sample_query_input):
    """Test the specified fields for each document are returned as a concatenated string."""

    embedded_fields = ["job_title", "employer"]

    expected_result = [
        "statistician ONS",
        "economist HMT",
        "social researcher",
        "",
    ]

    assert (
        EmbeddingHandler._create_query_texts(
            sample_query_input, embedded_fields
        )
        == expected_result
    )


def test_create_query_texts_no_embedded_fields(sample_query_input):
    """Test an empty string is returned when no embedded fields are specified."""

    embedded_fields = []

    expected_result = ["", "", "", ""]

    assert (
        EmbeddingHandler._create_query_texts(
            sample_query_input, embedded_fields
        )
        == expected_result
    )


def test_process_result_for_rag():
    """Test the results from from retrieval are formatted correctly for RAG."""
    result = {
        "metadatas": [
            [
                {"label": "SOC1", "description": "SOC1 Description 1"},
                {"label": "SOC2", "description": "SOC2 Description 2"},
            ],
            [
                {"label": "SOC3", "description": "SOC3 Description 3"},
                {"label": "SOC4", "description": "SOC4 Description 4"},
            ],
        ]
    }

    expected_result = [
        "SOC1:SOC1 Description 1\nSOC2:SOC2 Description 2",
        "SOC3:SOC3 Description 3\nSOC4:SOC4 Description 4",
    ]

    assert EmbeddingHandler.process_result_for_rag(result) == expected_result
