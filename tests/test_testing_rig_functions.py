"""Adding tests for testing rig."""

from unittest import mock

import pandas as pd
import pytest

from classifai.testing_rig_functions import (
    add_llm_results_to_table,
    add_retrieval_results_to_table,
    create_result_table,
    read_csv_to_dict_list,
)

"""Tests for read_csv_to_dict_list."""


@mock.patch("builtins.open")
def test_read_csv_to_dict_list(mock_open):
    """
    Test the read_csv_to_dict_list function.

    This test case mocks the CSV file data and asserts that the function
    correctly reads the CSV file and returns a list of dictionaries.
    """
    # Mock the CSV file data
    csv_data = [
        "name,age",
        "John,25",
        "Jane,30",
    ]
    mock_file = mock_open.return_value.__enter__.return_value
    mock_file.__iter__.return_value = csv_data

    # Call the function
    result = read_csv_to_dict_list("test.csv")

    # Assert the result
    assert result == [
        {"name": "John", "age": "25"},
        {"name": "Jane", "age": "30"},
    ]


@mock.patch("builtins.open")
def test_read_csv_to_dict_list_empty_file(mock_open):
    """
    Test the test_read_csv_to_dict_list_empty_file function.

    This test case mocks any empty CSV file data and asserts that the
    function correctly returns an empty list.
    """

    # Mock an empty CSV file
    mock_file = mock_open.return_value.__enter__.return_value
    mock_file.__iter__.return_value = []

    # Call the function
    result = read_csv_to_dict_list("test.csv")

    # Assert the result
    assert result == []


@mock.patch("builtins.open")
def test_read_csv_to_dict_list_file_not_found(mock_open):
    """
    Test case for the read_csv_to_dict_list function when the file is not found.

    This test case mocks a file not found error by setting the side_effect of the
    mock_open function to FileNotFoundError. It then calls the read_csv_to_dict_list
    function with a non-existent file name and expects it to raise a FileNotFoundError.
    """

    # Mock a file not found error
    mock_open.side_effect = FileNotFoundError

    # Call the function
    with pytest.raises(FileNotFoundError):
        read_csv_to_dict_list("test.csv")


"""Tests for create_result_table."""


@mock.patch("pandas.read_csv")
def test_create_result_table(mock_read_csv):
    """
    Test the create_result_table function.

    This test case mocks the input data file and asserts that the function
    correctly creates a blank result table with the expected columns.
    """

    # Mock the input data file
    input_data_filepath = "test_data.csv"
    mock_read_csv.return_value = pd.DataFrame()

    # Call the function
    result_table = create_result_table(
        {"number_of_retrievals": 1}, input_data_filepath
    )

    # Assert the result
    assert isinstance(result_table, pd.DataFrame)
    assert "top_retrieval_accuracy" in result_table.columns
    assert "top_1_retrievals_accuracy" in result_table.columns
    assert "top_llm_accuracy" in result_table.columns
    assert "top_llm_confidence" in result_table.columns


def test_add_retrieval_results_to_table():
    """
    Test function for adding retrieval results to a table.

    This function creates a sample result table and a sample processed result.
    It then calls the `add_retrieval_results_to_table` function with the sample data.
    Finally, it asserts the correctness of the updated result table.
    """

    result_table = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "label_column_dataset": [1, 2, 3],
            "top_retrieval_accuracy": [0, 0, 0],
            "top_2_retrievals_accuracy": [0, 0, 0],
        }
    )

    # Create a sample processed result
    processed_result = {
        "1": [{"label": 1, "distance": 5}, {"label": 2, "distance": 3}],
        "2": [{"label": 1, "distance": 2}, {"label": 2, "distance": 1}],
        "3": [{"label": 4, "distance": 6}, {"label": 5, "distance": 7}],
    }

    # Call the function
    updated_result_table = add_retrieval_results_to_table(
        {
            "id_column_dataset": "id",
            "label_column_dataset": "label_column_dataset",
            "number_of_retrievals": 2,
            "number_of_digit_classification": 4,
        },
        result_table,
        processed_result,
    )

    # Assert the updated result table
    assert updated_result_table["top_retrieval_accuracy"].tolist() == [1, 0, 0]
    assert updated_result_table["top_2_retrievals_accuracy"].tolist() == [
        1,
        1,
        0,
    ]


"""Test add_llm_results_to_table."""


@mock.patch("pandas.DataFrame.to_csv")
def test_add_llm_results_to_table(mock_to_csv):
    """
    Test the add_llm_results_to_table function.

    This test case mocks the llm_result_dict and folder_name parameters.
    It asserts that the function correctly updates the result_table and saves the results table.
    """

    # Mock the llm_result_dict and folder_name parameters
    llm_result_dict = {
        "1": [{"1": 90}],
        "2": [{"2": 80}],
        "3": [{"2": 70, "3": 55}],
    }
    folder_name = "test_folder"

    # Create a sample result_table
    result_table = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "label_column_dataset": [1, 2, 3],
            "top_llm_accuracy": [0, 0, 0],
            "top_llm_confidence": [0, 0, 0],
        }
    )

    # Call the function
    updated_result_table = add_llm_results_to_table(
        {
            "id_column_dataset": "id",
            "label_column_dataset": "label_column_dataset",
            "number_of_digit_classification": 4,
        },
        result_table,
        llm_result_dict,
        folder_name,
    )

    # Assert the updated result_table
    assert updated_result_table["top_llm_accuracy"].tolist() == [1, 1, 0]
    assert updated_result_table["top_llm_confidence"].tolist() == [90, 80, 70]

    # Assert that the results table is saved to CSV
    mock_to_csv.assert_called_once_with(f"{folder_name}/accuracy.csv")


@mock.patch("pandas.DataFrame.to_csv")
def test_add_llm_results_to_table_empty_result_table(mock_to_csv):
    """
    Test the add_llm_results_to_table function with an empty result_table.

    This test case asserts that the function returns an empty result_table when the input result_table is empty.
    """

    # Create an empty result_table
    result_table = pd.DataFrame()

    # Call the function
    updated_result_table = add_llm_results_to_table(
        {
            "id_column_dataset": "id",
            "label_column_dataset": "label_column_dataset",
        },
        result_table,
        {},
        "test_folder",
    )

    # Assert that the updated result_table is empty
    assert updated_result_table.empty
