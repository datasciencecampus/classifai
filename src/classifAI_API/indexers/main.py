"""This module provides functionality for creating a vector index from a text file
and saving the processed data as a Parquet file. It is designed to handle large
datasets by processing the input file in batches, extracting text data, generating
vector embeddings, and storing the results in a structured format.

The main function in this module, `create_vector_index_from_string_file`, supports
CSV files as input and uses a provided embedder to generate vector embeddings for
the text data. The processed data includes unique identifiers, text content, and
their corresponding embeddings, which are saved in a Parquet file for efficient
storage and retrieval.

Key Features:
- Batch processing of input files to handle large datasets.
- Support for CSV file format (additional formats may be added in future updates).
- Integration with a custom embedder for generating vector embeddings.
- Logging for tracking progress and handling errors during processing.

Dependencies:
- pandas: For handling data in tabular format and saving it as a Parquet file.
- tqdm: For displaying progress bars during batch processing.
- A custom file iterator (`iter_csv`) for reading input files in batches.

Usage:
This module is intended to be used as part of a larger application that requires
vector indexing of text data. It can be extended to support additional file formats
and embedding methods as needed.
"""

import logging

import polars as pl
import tqdm

from .helpers.file_iters import iter_csv

# Configure logging for your application
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


def create_vector_index_from_string_file(
    file_name,
    data_type,
    meta_data,
    embedder,
    batch_size,
):
    """Creates a vector index from a file containing text data and saves it as a Parquet file.
    This function processes a file in batches, extracts text data, generates vector embeddings
    using the provided embedder, and saves the resulting data (IDs, text, and embeddings)
    into a Parquet file.

    Args:
        file_name (str): The path to the input file containing the text data.
        data_type (str): The type of the input file. Currently, only 'csv' is supported.
        meta_data (list): A list additional columns to include in the output DataFrame, other than ['id', 'text'].
        embedder: An instance of the `Vectoriser` class from the
            `vectorisers` module, used to generate vector embeddings for the text data.
        batch_size (int): The number of rows to process in each batch.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed data with columns:
            - 'id': The unique identifier for each row.
            - 'text': The text data.
            - 'embeddings': The vector embeddings generated for the text data.

    Raises:
        Exception: If an unsupported file type is provided or if there are errors during
            file processing, vectorization, or saving the Parquet file.

    Notes:
        - The function currently supports only CSV files. Additional file types may be
        supported in future updates.
        - The Parquet file is saved with the same name as the input file, but with a
        `.parquet` extension.
    """
    # set up the file indexer
    try:
        if data_type == "csv":
            file_loader = iter_csv

        else:
            raise Exception(
                "FileType must be of type string and one of ['csv']   (more file types added in later update!)"
            )

    except Exception:
        logging.error("Error setting up file loader: {e}")
        raise

    # Process the file in batches
    captured_data = {x: [] for x in ["id", "text", *meta_data]}
    captured_embeddings = []

    logging.info(
        "Processing file: %s in batches of size %d...\n", file_name, batch_size
    )
    for batch_no, batch in enumerate(
        tqdm.tqdm(
            file_loader(
                file_name,
                meta_data=meta_data,
                batch_size=batch_size,
            ),
            desc="Processing batches",
        )
    ):
        # try:
        # Extract IDs and texts
        for k in captured_data.keys():

            captured_data[k].extend([entry[k] for entry in batch])

        # Send the batch to be vectorized
        batch_vectors = embedder.transform([entry["text"] for entry in batch])
        captured_embeddings.extend(batch_vectors)

        # except Exception as e:
        # logging.error(f"Error processing batch {batch_no}: {e}")
        # continue

    print("---------")
    logging.info("Finished creating vectors, attempting to save to parquet file...")

    # finally collect all the data in polars dataframe and save it to a parquet file
    try:
        df = pl.DataFrame({x: captured_data[x] for x in captured_data})
        df = df.with_columns(pl.Series("embeddings", captured_embeddings))
        df.write_parquet(f"{file_name.replace('.csv', '.parquet')}")

    except Exception as e:
        logging.error(f"Error creating Polars DataFrame or saving to Parquet file: {e}")
        raise

    logging.info(
        f"DataFrame created with {len(df)} rows and {len(df.columns)} columns."
    )
    logging.info(f"Saved DataFrame to Parquet file: {file_name}.parquet")

    return df
