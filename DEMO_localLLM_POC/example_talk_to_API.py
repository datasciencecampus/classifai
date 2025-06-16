#!/usr/bin/env python
import requests
import json

endpoint_base_url = "127.0.0.1"
endpoint_port = "8000"

with open('./test_data.json', 'r') as f:
    query_data = json.load(f)

# Using the /search endpoint provided with the app 
search_response = requests.post(f"http://{endpoint_base_url}:{endpoint_port}/search?n_results=3", 
                               json=query_data)
search_data = json.loads(search_response.content.decode('utf-8'))

print(f'Using the /search endpoint, the output is:\n')
for idx,reply in enumerate(search_data['data']):
    print(f'For input "{query_data['entries'][idx]['description']}", the top 3 categories are:')
    for guess in reply['response']:
        print(f"   {guess['rank']} (distance {guess['distance']:.4f}): {guess['description']}")
    print('\n')