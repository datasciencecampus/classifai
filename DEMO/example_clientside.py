#!/usr/bin/env python
import requests
import json
import numpy as np

endpoint_base_url = "127.0.0.1"
endpoint_port = "8000"

def get_idx_for_k_best(k, scores):
    return np.argpartition(scores, -k)[-k:][::-1]

with open('./test_data.json', 'r') as f:
    query_data = json.load(f)

# First request to server - using the /embed endpoint provided with app
embed_response = requests.post(f"http://{endpoint_base_url}:{endpoint_port}/embed", 
                         json=query_data)

embed_response_dict = json.loads(embed_response.content.decode('utf-8'))

# Requests to newly made /new_endpoint_score endpoint to score each possible classification 
for idx,qe in enumerate(embed_response_dict['data']):
    score_response = requests.post(f"http://{endpoint_base_url}:{endpoint_port}/new_endpoint_score", 
                         json=qe['embedding'])
    embed_response_dict['data'][idx]['score'] = np.array(json.loads(score_response.content))

print(f'Using the provided /embed endpoint and the supplied /new_endpoint_score endpoint, the output is:\n')

# Client-side ranking and sorting of classifications for each input based on scores
for idx,qe in enumerate(embed_response_dict['data']):
    print(f'For input "{qe['description']}", the top 3 categories are:')
    top_3_ids = get_idx_for_k_best(3, qe['score'])
    for rank, rank_idx in enumerate(top_3_ids):
        distance = 2*(1 - qe['score'][rank_idx])
        print(f"   {rank+1} (distance {distance:.4f}): {embed_response_dict['category_labels'][rank_idx]}")
    print('\n')

# Using the original classifAI /soc endpoint for validation 
soc_response = requests.post(f"http://{endpoint_base_url}:{endpoint_port}/search?n_results=3", 
                               json=query_data)

original_soc_data = json.loads(soc_response.content)

print(f'Using the original /soc endpoint, the output is:\n')
for idx,reply in enumerate(original_soc_data['data']):
    print(f'For input "{query_data['entries'][idx]['description']}", the top 3 categories are:')
    for guess in reply['response']:
        print(f"   {guess['rank']} (distance {guess['distance']:.4f}): {guess['description']}")
    print('\n')