"""Tests for API class."""

from classifai import API

tool = API(input_filepath="tests/data/lfs_test.csv")


def test_jsonify_input():
    """Checks input data transformed to dictionary."""

    test_json = tool.jsonify_input()
    assert len(test_json) == 1
    assert test_json[0]["job_title"] == "Musician"


def test_classify_input():
    """Checks classified data transformed correctly."""

    test_json = [
        {
            "uid": "0023",
            "job_title": "lecturer",
            "job_description": "",
            "level_of_education": "",
            "manage_others": "",
            "industry_descr": "",
            "miscellaneous": "",
        }
    ]
    test_json = tool.classify_input(test_json)
    assert list(test_json[0].keys()) == [
        "uid",
        "job_title",
        "job_description",
        "level_of_education",
        "manage_others",
        "industry_descr",
        "miscellaneous",
        "label",
        "distance",
    ]


def test_simplify_output(
    sample_query_input, sample_query_result, sample_query_processed
):
    """Test the output from the is processed to a JSON correctly."""

    assert (
        API.simplify_output(sample_query_result, sample_query_input[0:2], "id")
        == sample_query_processed
    )
