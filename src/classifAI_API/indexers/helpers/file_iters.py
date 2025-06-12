"""This module contains functions for ways of iterating through data efficiently using generators.
Each function in this module is designed to yield batches as lists of dictionaries, where each
dictionary contains 'id' and 'text' keys with their values. Currently, the module includes
an implementation for iterating through a CSV file.

Functions:
    - iter_csv: Reads a CSV file in batches and yields dictionaries containing rows of data.

    Reads a CSV file in batches and yields dictionaries containing rows of data.
"""


def iter_csv(file_name, batch_size=8):
    """Reads a CSV file in batches and yields dictionaries containing rows of data.
    This function reads a CSV file line by line, validates the header row, and
    yields batches of rows as dictionaries. Each dictionary contains the 'id'
    and 'text' fields from the CSV file. The function skips invalid rows and
    ensures the header matches the expected format.

    Args:
        file_name (str): The path to the CSV file to be read.
        batch_size (int, optional): The number of rows to include in each batch.
            Defaults to 8.

    Yields:
        list[dict]: A batch of rows, where each row is represented as a dictionary
            with 'id' and 'text' keys.

    Raises:
        ValueError: If the CSV file does not have a header row with 'id' and 'text'
            columns.
    """
    # Open the CSV file
    with open(file_name, encoding="utf-8") as file:
        # Validate the header row
        header = file.readline().strip().split(",")
        if header != ["id", "text"]:
            raise ValueError(
                "CSV file must have a header row with 'id' and 'text' columns"
            )

        # Read the file in chunks, discarding the header
        batch = []
        for line in file:

            row = line.strip().split(",")
            if len(row) != 2:
                continue  # Skip invalid rows
            batch.append({"id": row[0], "text": row[1]})
            if len(batch) == batch_size:
                yield batch
                batch = []

        # Yield any remaining rows
        if batch:
            yield batch
