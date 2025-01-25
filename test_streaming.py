import requests
import json

url = "http://localhost:8500/v2/models/kazllm/generate_stream"
payload = {
    "texts": "What is machine learning?",
    "task": "",
    "question": ""
}


response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        json_response = json.loads(line.decode('utf-8').lstrip('data: '))
        print(json_response['output'], end='', flush=True)
