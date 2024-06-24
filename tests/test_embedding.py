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


def test_process_output(
    sample_query_input, sample_query_result, sample_query_processed
):
    """Test the output from the is processed to a JSON correctly."""

    assert (
        EmbeddingHandler._process_output(
            sample_query_result, sample_query_input[0:2], "id"
        )
        == sample_query_processed
    )
