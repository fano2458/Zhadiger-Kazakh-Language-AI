import requests
import json
import base64


url = "http://localhost:8500/v2/models/ensemble_ocr_rag_kazllm/generate_stream"



with open("НУ ДБҰ магистратура қағидалары.pdf", "rb") as image_file:
    pdf1_bytes = image_file.read()
    pdf1_encoded = base64.b64encode(pdf1_bytes).decode('utf-8')

with open("НУ_бакалавриат_қабылдау_қағидалары-1-2.pdf", "rb") as image_file:
    pdf2_bytes = image_file.read()
    pdf2_encoded = base64.b64encode(pdf2_bytes).decode('utf-8')

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
    "texts": [pdf1_encoded],  # OCR expects file paths or base64 images
    "user_request": ["магистр дәрежесін АЛУ ҮШІН IELTS-тен қанша балл жинау керек?"],  # Your user query
    "task": "qa"  # Task type for the model
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        json_response = json.loads(line.decode('utf-8').lstrip('data: '))
        # pri
        print(json_response['output'], end='', flush=True)