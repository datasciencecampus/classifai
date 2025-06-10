import logging
import tqdm 
import pandas as pd
from .helpers.file_iters import iter_csv

# Configure logging for your application
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)



def create_vector_index_from_string_file(
    fileName, 
    dataType,
    embedder,
    batch_size,):


    #set up the file indexer
    try:
        if dataType == 'csv':
            file_loader = iter_csv
        
        else:
            raise "FileType must be of type string and one of ['csv']   (more file types added in later update!)"

    except Exception as e:
        logging.error("Error setting up file loader: {e}")
        raise



    # Process the file in batches
    ids = []
    texts = []
    vectors = []

    logging.info(f"Processing file: {fileName} in batches of size {batch_size}...\n")
    for batch_no, batch in enumerate(tqdm.tqdm(file_loader(fileName, batch_size=batch_size,), desc="Processing batches")):
        try:
            # Extract IDs and texts
            batch_ids = [entry['id'] for entry in batch]
            batch_texts = [entry['text'] for entry in batch]

            # Send the batch to be vectorized
            batch_vectors = embedder.transform(batch_texts)

            # Store the ids, text, and vectors in their corresponding lists
            ids.extend(batch_ids)
            texts.extend(batch_texts)
            vectors.extend(batch_vectors)

        except Exception as e:
            logging.error(f"Error processing batch {batch_no}: {e}")
            continue
    
    print("---------")
    logging.info(f"Finished creating vectors, attempting to save to parquet file...")


    #finally collect all the data in pandas dataframe and save it to a parquay file
    try:
        df = pd.DataFrame({'id': ids, 'text': texts, 'embeddings': vectors})
        df.to_parquet(f"{fileName.split('.')[0]}.parquet")

    except Exception as e:
        logging.error(f"Error creating DataFrame or saving to Parquet file: {e}")
        raise
    

    logging.info(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns.")
    logging.info(f"Saved DataFrame to Parquet file: {fileName}.parquet")


    return df

