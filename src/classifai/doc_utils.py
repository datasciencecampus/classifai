"""Utility functions for the Quarto tutorials."""

import re


def clean_text(text: str) -> str:
    """Clean text data for exact matching.

    For example:
    "Teacher, Statistics (secondary school)" -> "secondary school statistics teacher"

    Parameters
    ----------
    text : str
        Unprocessed text data.

    Returns
    -------
    text: str
        Cleaned text data for exact matching.
    """
    if text == "":
        return ""

    if "(" in text:
        text_in_brackets = text.split("(", 1)[1].split(")")[0]
        text_in_brackets_processed = text_in_brackets.replace(":", "")
        text_in_brackets_processed = text_in_brackets_processed.replace(
            ",", "or"
        )
        text_without_brackets = text.replace(f"({text_in_brackets})", "")
    else:
        text_in_brackets_processed = ""
        text_without_brackets = text

    text_without_brackets_processed = text_without_brackets.split(", ")
    text_without_brackets_processed
    text_without_brackets_processed = " ".join(
        text_without_brackets_processed[::-1]
    )

    text_processed = (
        f"{text_in_brackets_processed} {text_without_brackets_processed}"
    )
    text_processed = re.sub(" +", " ", text_processed)
    text_processed = text_processed.strip().lower()

    return text_processed


def clean_job_title(job_title: str) -> str:
    """Remove the initialism `n.e.c.` from job titles.

    This is for jobs with SOC codes ending in `99` for 6-digit SOC
    codes.

    Parameters
    ----------
    job_title : str
        The job title to be cleaned.

    Returns
    -------
    str
        Job title free of the initialism.
    """
    return re.sub("n.e.c.", "", job_title).strip()


def clean_job_description(job_title: str, job_description: str) -> str:
    """Replace the job description with the job title for 6-digit SOC codes ending in  `99`.

    Parameters
    ----------
    job_title : str
        The job title for a 6-digit SOC code.
    job_description : str
        The job description for a 6-digit SOC code.

    Returns
    -------
    str
        The current job description for all jobs with a SOC code not ending in `99` and
        the job title for jobs with a SOC code ending in `99`.
    """
    if "Job holders in this group perform" in job_description:
        return job_title
    else:
        return job_description


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
