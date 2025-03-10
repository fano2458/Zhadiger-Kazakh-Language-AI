import requests
import json
import base64


url = "http://localhost:8500/v2/models/ensemble_ocr_rag_kazllm/generate_stream"

with open("img1.png", "rb") as image_file:
    img1_bytes = image_file.read()
    img1_encoded = base64.b64encode(img1_bytes).decode('utf-8')

with open("img2.png", "rb") as image_file:
    img2_bytes = image_file.read()
    img2_encoded = base64.b64encode(img2_bytes).decode('utf-8')

with open("img3.png", "rb") as image_file:
    img3_bytes = image_file.read()
    img3_encoded = base64.b64encode(img3_bytes).decode('utf-8')


payload = {
    "texts": [img1_encoded, img2_encoded, img3_encoded],  # OCR expects file paths or base64 images
    "user_request": ["Нұрсұлтан Назарбаев неше жаста?"],  # Your user query
    "file_paths": ["img1", "img2", "img3"],  # For RAG
    "task": "qa"  # Task type for the model
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        json_response = json.loads(line.decode('utf-8').lstrip('data: '))
        print(json_response['output'], end='', flush=True)