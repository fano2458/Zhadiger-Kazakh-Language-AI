import requests
import json
import time


def get_embeddings(text):
    url = "http://localhost:8500/v2/models/kazllm/infer"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": [
            {
                "name": "TEXTS",  
                "shape": [len(text)], 
                "datatype": "BYTES",
                "data": text
            }
        ]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        response_data = response.json()
        embeddings = response_data['outputs'][0]['data']
        return embeddings
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


text = ["қазақстанның астанасы қандай?"] 
start_time = time.time()
embeddings = get_embeddings(text)
print(embeddings)
print(f"Total time is {time.time() - start_time}")
# print(len(embeddings))