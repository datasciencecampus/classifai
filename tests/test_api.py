"""Tests for API class."""

import pytest

from classifai import API

tool = API()


def test_jsonify_input(survey_csv):
    """Checks input data transformed to dictionary."""

    tool.input_filepath = survey_csv
    test_json = tool.jsonify_input()
    assert len(test_json) == 1
    assert test_json[0]["job_title"] == "Musician"


@pytest.mark.skip(
    "Disable temporarily - will use FakeEmbeddings to facilitate a temp vector store"
)
def test_classify_input(sample_query_input):
    """Checks classified data transformed correctly."""

    classification = tool.classify_input(
        input_data=sample_query_input, embedded_fields=["job_title", "company"]
    )

    assert list(classification.keys()) == [
        "ids",
        "distances",
        "metadatas",
        "embeddings",
        "documents",
        "uris",
        "data",
        "included",
    ]


def test_simplify_output(
    sample_query_input, sample_query_result, sample_query_processed
):
    """Test the output from the query is processed to a JSON correctly."""

    assert (
        API.simplify_output(sample_query_result, sample_query_input[0:2], "id")
        == sample_query_processed
    )
