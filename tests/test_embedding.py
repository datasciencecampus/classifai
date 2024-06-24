"""Tests the EmbeddingHandler class."""

from classifai.embedding import EmbeddingHandler


def test_create_query_texts():
    """Test the specified fields for each document are returned as a concatenated string."""

    input_data = [
        {"id": 1, "job_title": "statistician", "employer": "ONS"},
        {"id": 2, "job_title": "economist", "employer": "HMT"},
        {"id": 3, "job_title": "social researcher", "description": "Analyst."},
        {},
    ]

    embedded_fields = ["job_title", "employer"]

    expected_result = [
        "statistician ONS",
        "economist HMT",
        "social researcher",
        "",
    ]

    assert (
        EmbeddingHandler._create_query_texts(input_data, embedded_fields)
        == expected_result
    )


def test_create_query_texts_no_embedded_fields():
    """Test an empty string is returned when no embedded fields are specified."""

    input_data = [{"id": 1, "job_title": "statistician", "employer": "ONS"}]

    embedded_fields = []

    expected_result = [""]

    assert (
        EmbeddingHandler._create_query_texts(input_data, embedded_fields)
        == expected_result
    )


# def test_process_output():
