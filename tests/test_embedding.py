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


def test_process_output():
    """Test the output from the is processed to a JSOn correctly."""

    query_result = {
        "ids": [["a", "b"], ["c"]],
        "distances": [[13.2, 12], [10, 9]],
        "metadatas": [
            [{"label": 212}, {"label": 12}],
            [{"label": 34}, {"label": 25}],
        ],
        "embeddings": None,
        "documents": [["Job 1", "Job 2"], ["Job 3", "Job 4"]],
    }

    input_data = [{"id": "1"}, {"id": 2}]

    expected_output = {
        "1": [
            {"label": 212, "description": "Job 1", "distance": 13.2},
            {"label": 12, "description": "Job 2", "distance": 12},
        ],
        2: [
            {"label": 34, "description": "Job 3", "distance": 10},
            {"label": 25, "description": "Job 4", "distance": 9},
        ],
    }

    assert (
        EmbeddingHandler._process_output(query_result, input_data, "id")
        == expected_output
    )
