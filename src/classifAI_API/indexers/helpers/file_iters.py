import pandas as pd

def iter_csv(fileName, batch_size=8):
    # Open the CSV file
    with open(fileName, 'r') as file:
        # Validate the header row
        header = file.readline().strip().split(',')
        if header != ['id', 'text']:
            raise ValueError("CSV file must have a header row with 'id' and 'text' columns")

        # Read the file in chunks, discarding the header
        batch = []
        for line in file:
            
            row = line.strip().split(',')
            if len(row) != 2:
                continue  # Skip invalid rows
            batch.append({'id': row[0], 'text': row[1]})
            if len(batch) == batch_size:
                yield batch
                batch = []

        # Yield any remaining rows
        if batch:
            yield batch