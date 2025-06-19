"""This module contains functions for ways of iterating through data efficiently using generators.
Each function in this module is designed to yield batches as lists of dictionaries, where each
dictionary contains 'id' and 'text' keys with their values. Currently, the module includes
an implementation for iterating through a CSV file.

Functions:
    - iter_csv: Reads a CSV file in batches and yields dictionaries containing rows of data.

    Reads a CSV file in batches and yields dictionaries containing rows of data.
"""


def iter_csv(file_name, meta_data=None, batch_size=8):
    """Reads a CSV file in batches and yields dictionaries containing rows of data.
    This function reads a CSV file line by line, validates the header row, and
    yields batches of rows as dictionaries. Each dictionary contains all fields
    from the CSV file, with keys corresponding to the header names.

    Args:
        file_name (str): The path to the CSV file to be read.
        meta_data (list): A list of additional columns to include in the output,
        batch_size (int, optional): The number of rows to include in each batch.
            Defaults to 8.

    Yields:
        list[dict]: A batch of rows, where each row is represented as a dictionary
            with keys corresponding to the header names.

    Raises:
        ValueError: If the CSV file does not have a valid header row.
    """
    if meta_data is None:
        meta_data = []

    # combine metadata and text and id column parameters
    columns = ["id", "text", *meta_data]

    # Open the CSV file
    with open(file_name, encoding="utf-8") as file:

        # Read and validate the header row
        headers = file.readline().strip().split(",")

        if not headers:
            raise ValueError("CSV file must have a valid header row")

        # Check if each row in the columns that we want to collect, is in fact in the csv file headers
        for each in columns:
            if each not in headers:
                raise ValueError(f"CSV file header is missing required column: {each}")

        # Read the file in chunks, discarding the header
        batch = []
        for line in file:
            row = line.strip().split(",")
            if len(row) != len(headers):
                continue  # Skip invalid rows
            batch.append(dict(zip(headers, row, strict=False)))
            if len(batch) == batch_size:
                yield batch
                batch = []

        # Yield any remaining rows
        if batch:
            yield batch
