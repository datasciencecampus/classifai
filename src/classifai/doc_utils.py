"""Utility functions for the Quarto tutorials."""


def print_processed_output(
    processed_result: dict, input_data: list[dict], embedded_fields: list
):
    """Print the output from API.simplify_output and the input data.

    Parameters
    ----------
    processed_result : dict
        The output from API.simplify_output.
    input_data : list[dict]
        List of dictionaries of input survey data.
    embedded_fields : list
        The list of fields embedded and searched against the database.
    """

    for u, input_document in zip(processed_result.values(), input_data):
        print("Input:")
        for v, w in input_document.items():
            if v in embedded_fields:
                print(f"  {w}")
        print("")
        for index, result_dictionary in enumerate(u):
            print(f"Search Result {index}")
            for x, y in result_dictionary.items():
                print(f"  {x}: {y}")
        print("")
        print("===========================================")
        print("")
