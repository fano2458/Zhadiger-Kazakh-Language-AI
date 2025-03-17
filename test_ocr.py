import requests
import json
import base64


with open("НУ ДБҰ магистратура қағидалары-1-2-1.pdf", "rb") as image_file:
    pdf1_bytes = image_file.read()
    pdf1_encoded = base64.b64encode(pdf1_bytes).decode('utf-8')

with open("НУ ДБҰ магистратура қағидалары-1-2.pdf", "rb") as image_file:
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


images = [img1_encoded, img2_encoded, img3_encoded]
pdfs = [pdf1_encoded, pdf2_encoded]

payload = {
            "inputs": [
                {
                    "name": "texts",
                    "shape": [len(pdfs)],
                    "datatype": "BYTES",
                    "data": pdfs
                }
            ]
        }

url = f"http://localhost:8500/v2/models/ocr/infer"
headers = {
        'Content-Type': 'application/json',
    }

response = requests.post(url, headers=headers, data=json.dumps(payload))


if response.status_code == 200:
    response_data = response.json()

    response_result = response_data['outputs']
    
    for res in response_result:
        for el in res["data"]:
            print(el)
else:
    print(response)
