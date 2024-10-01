"""Functions to use as part of the testing rig."""

import csv
import json
import os
from datetime import datetime

import pandas as pd

from classifai.api import API
from classifai.embedding import EmbeddingHandler
from classifai.llm import ClassificationLLM


def read_csv_to_dict_list(file_path: str) -> list[dict]:
    """Read in the CSV and convert to a list of dictionaries.

    Parameters
    ----------
    file_path : str
        Relative filepath to the test data.

    Returns
    -------
    dict_list : list[dict]
        The test data with each row represented by a dictionary.
    """

    with open(file_path, mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        dict_list = [row for row in reader]
    return dict_list


def create_result_table(
    test_parameters: dict, input_data_filepath: str
) -> pd.DataFrame:
    """Create a blank result table.

    The table records the following metrics:
    'top_retrieval_accuracy' : 1 if the closest SOC code from the retrieval step is a true positive.
    'top_retrieval_distance' : The distance between the embedding of the entry and the closest SOC code.
    'top_`k`_retrievals_accuracy' : 1 if any of the SOC codes from the retrieval step is a true positive.
    'top_llm_accuracy' : 1 if the SOC code from the LLM with the highest confidence score is a true positive.
    'top_llm_confidence' : The highest confidence score of any of the SOC codes provided by the LLM. Between 0 and 100.

    Parameters
    ----------
    test_parameters : dict
        Dictionary of test parameters.
    input_data_filepath : str
        Relative filepath to the test data.

    Returns
    -------
    result_table: pd.DataFrame
        Results table. All of the metrics are set to 0 by default.
    """

    result_table = pd.read_csv(input_data_filepath)
    result_table[
        [
            "top_retrieval_accuracy",
            "top_retrieval_distance",
            f"top_{test_parameters['number_of_retrievals']}_retrievals_accuracy",
            "top_llm_accuracy",
            "top_llm_confidence",
        ]
    ] = 0
    return result_table


def create_output_folder(test_parameters: dict) -> str:
    """Create a folder for the outputs.

    Parameters
    ----------
    test_parameters : dict
        Dictionary of test parameters.

    Returns
    -------
    folder_name : str
        The folder name to store the results of the test.
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    folder_name = (
        f"outputs/{formatted_datetime}_{test_parameters['test_name']}"
    )
    os.mkdir(folder_name)

    with open(f"{folder_name}/metadata.json", "w") as outfile:
        json.dump(test_parameters, outfile)

    return folder_name


def create_embedding_index(test_parameters: dict) -> EmbeddingHandler:
    """Embed the index and return an EmbeddingHandler object.

    Parameters
    ----------
    test_parameters : dict
        Dictionary of test parameters.

    Returns
    -------
    embed : EmbeddingHandler
        EmbeddingHandler class from classifai.embedding.
    """
    embed = EmbeddingHandler(
        embedding_model_name=test_parameters["embedding_model"],
        k_matches=test_parameters["number_of_retrievals"],
        distance_metric=test_parameters["distance_metric"],
        create_vector_store=True,
    )

    try:
        if test_parameters["classification_type"] == "soc":
            file_name = (
                f"data/soc-index/{test_parameters['embedding_index_file']}"
            )
        else:
            file_name = (
                f"data/sic-index/{test_parameters['embedding_index_file']}"
            )

        if file_name[-3:] == "csv":
            embed.embed_index_csv(
                file=file_name,
                label_column=test_parameters["label_column_index"],
                embedding_columns=test_parameters["embedding_columns_index"],
            )

        elif test_parameters["classification_type"] == "soc":
            embed.embed_master_index(file_name=file_name)
        else:
            embed.embed_index(file_name=file_name)

        return embed
    except ValueError:
        print(
            "Maximum number of calls to API reached. Terminating embedding procedure."
        )


def search_embedding_index(
    test_parameters: dict,
    input_data: list[dict],
    embed: EmbeddingHandler,
    folder_name: str,
) -> tuple[list[dict], dict]:
    """Search the embedding index and process the results.

    Parameters
    ----------
    test_parameters : dict
        Dictionary of test parameters.
    input_data: list[dict]
        List of dictionaries of the test data.
    embed : EmbeddingHandler
        EmbeddingHandler class from classifai.embedding.
    folder_name : str
        The folder name to store the results of the test.

    Returns
    -------
    result : list[dict]
        The raw result from the embedding search.
    processed_result : dict
        The processed result from the embedding search.
    """
    result = embed.search_index(
        input_data=input_data,
        embedded_fields=test_parameters["embedding_columns_dataset"],
    )

    processed_result = API.simplify_output(
        output_data=result, input_data=input_data, id_field="id"
    )

    # Save the results to disk
    with open(f"{folder_name}/processed_retrieval.json", "w") as outfile:
        json.dump(processed_result, outfile)
    return result, processed_result


def add_retrieval_results_to_table(
    test_parameters: dict, result_table: pd.DataFrame, processed_result: dict
) -> pd.DataFrame:
    """Add the accuracy results from the retrieval to the results table.

    Parameters
    ----------
    test_parameters : dict
        Dictionary of test parameters.
    result_table : pd.DataFrame
        Table of results.
    processed_result : dict
        The processed results from the retrieval.

    Returns
    -------
    result_table : pd.DataFrame
        Updated result_table.
    """
    for index in range(len(result_table)):
        id = result_table.loc[index, test_parameters["id_column_dataset"]]

        # top retrieval accuracy
        if (
            str(processed_result[str(id)][0]["label"])[
                0 : test_parameters["number_of_digit_classification"]
            ]
            == str(
                result_table.loc[
                    index, test_parameters["label_column_dataset"]
                ]
            )[0 : test_parameters["number_of_digit_classification"]]
        ):
            result_table.loc[index, "top_retrieval_accuracy"] = 1

        # top retrieval distance
        result_table.loc[index, "top_retrieval_distance"] = processed_result[
            str(id)
        ][0]["distance"]

        # top k retrieval accuracy
        for retrieval_result in processed_result[str(id)]:
            if (
                str(retrieval_result["label"])[
                    0 : test_parameters["number_of_digit_classification"]
                ]
                == str(
                    result_table.loc[
                        index, test_parameters["label_column_dataset"]
                    ]
                )[0 : test_parameters["number_of_digit_classification"]]
            ):
                result_table.loc[
                    index,
                    f"top_{test_parameters['number_of_retrievals']}_retrievals_accuracy",
                ] = 1

    return result_table


def set_up_llm(
    test_parameters: dict, embed: EmbeddingHandler, result: list[dict]
) -> tuple[list, ClassificationLLM]:
    """Set up the LLM.

    Parameters
    ----------
    test_parameters : dict
        Dictionary of test parameters.
    embed : EmbeddingHandler
        EmbeddingHandler class from classifai.embedding.
    result : list[dict]
        The raw result from the embedding search.

    Returns
    -------
    rag_candidates : list
        The list of candidate SOC codes for rag.

    classifier : ClassificationLLM
        Instance of the ClassificationLLM class.
    """
    rag_candidates = embed.process_result_for_rag(result)
    classifier = ClassificationLLM(model_name=test_parameters["llm"])
    return rag_candidates, classifier


def get_llm_results(
    test_parameters: dict,
    input_data: list[dict],
    rag_candidates: list,
    classifier: ClassificationLLM,
    folder_name: str,
) -> dict:
    """Get the results from the LLM.

    Parameters
    ----------
    test_parameters : dict
        Dictionary of test parameters.
    input_data: list[dict]
        List of dictionaries of the test data.
    rag_candidates : list
        The list of candidate SOC codes for rag.
    classifier : ClassificationLLM
        Instance of the ClassificationLLM class.
    folder_name : str
        The folder name to store the results of the test.

    Returns
    -------
    llm_result_dict : dict
        Processed dictionary of results from the LLM.
    """
    llm_result_dict = {}
    for input_document, search_result in zip(input_data, rag_candidates):
        result_list = []
        try:
            if test_parameters["classification_type"] == "soc":
                llm_result = classifier.get_soc_code(
                    input_document["job_title"],
                    None,
                    input_document["company"],
                    search_result,
                )
            else:
                llm_result = classifier.get_sic_code(
                    input_document["industry_descr"],
                    input_document["job_title"],
                    input_document["job_description"],
                    search_result,
                )
            for result in llm_result.candidates:
                result_list.append({result.code: result.likelihood})

        # Use bare except as the error type seems to change.
        except:  # noqa
            result_list.append({"Error": 0.0})

        llm_result_dict[input_document["id"]] = result_list

    # Save the results to disk
    with open(f"{folder_name}/processed_llm_results.json", "w") as outfile:
        json.dump(llm_result_dict, outfile)

    return llm_result_dict


def add_llm_results_to_table(
    test_parameters: dict,
    result_table: pd.DataFrame,
    llm_result_dict: dict,
    folder_name: str,
) -> pd.DataFrame:
    """Add results from the LLM classification to the results table.

    Save the results table.

    Parameters
    ----------
    test_parameters : dict
        Dictionary of test parameters.
    result_table : pd.DataFrame
        Table of results.
    llm_result_dict : dict
        Processed dictionary of results from the LLM.
    folder_name : str
        The folder name to store the results of the test.

    Returns
    -------
    result_table : pd.DataFrame
        Updated table of results.
    """
    for index in range(len(result_table)):
        id = result_table.loc[index, test_parameters["id_column_dataset"]]

        # top LLM accuracy
        if (
            str(list(llm_result_dict[str(id)][0].keys())[0])[
                0 : test_parameters["number_of_digit_classification"]
            ]
            == str(
                result_table.loc[
                    index, test_parameters["label_column_dataset"]
                ]
            )[0 : test_parameters["number_of_digit_classification"]]
        ):
            result_table.loc[index, "top_llm_accuracy"] = 1

        # top LLM confidence
        result_table.loc[index, "top_llm_confidence"] = list(
            llm_result_dict[str(id)][0].values()
        )[0]

    # Result table
    result_table.to_csv(f"{folder_name}/accuracy.csv")

    return result_table


def create_accuracy_summary_table(
    test_parameters: dict, result_table: pd.DataFrame, folder_name: str
):
    """Create a summary of the overall accuracy for retrieval and the LLM.

    Parameters
    ----------
    test_parameters : dict
        Dictionary of test parameters.
    result_table : pd.DataFrame
        Table of results.
    folder_name : str
        The folder name to store the results of the test.
    """

    accuracy_summary = (
        result_table[
            [
                "top_retrieval_accuracy",
                f"top_{test_parameters['number_of_retrievals']}_retrievals_accuracy",
                "top_llm_accuracy",
                "top_llm_confidence",
            ]
        ].sum(axis=0)
        / 20
    )

    # Save the results table to disk.
    accuracy_summary.to_csv(f"{folder_name}/accuracy_summary.csv")
