import requests
import json
import time
import io
from scipy.io import wavfile
import numpy as np
import base64


def get_embeddings(data, type="ner"):
    url = f"http://localhost:8500/v2/models/{type}/infer"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if type == "image_caption" or type == 'ocr':
        encoded_data = base64.b64encode(data).decode('utf-8')
        payload = {
            "inputs": [
                {
                    "name": "IMAGES",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [encoded_data]
                }
            ]
        }
    elif type == "tranlator":
        payload = {
            "inputs": [
                {
                    "name": "TEXTS",
                    "shape": [len(data)],
                    "datatype": "BYTES",
                    "data": data
                },
                {
                    "name": "LANG_TYPE",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": data
                },
                {
                    "name": "TRG_LANG",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": data
                }
            ]
        }
    else:
        payload = {
            "inputs": [
                {
                    "name": "TEXTS",  
                    "shape": [len(data)], 
                    "datatype": "BYTES",
                    "data": data
                }
            ]
        }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        response_data = response.json()
        response_result = response_data['outputs'][0]['data']

        if type == "ner":
            response_result = json.loads(response_result)
        elif type == "tts":
            response_result = np.array(response_result, dtype=np.uint8)
            with io.BytesIO(response_result) as wav_io:
                rate, data = wavfile.read(wav_io)
                return rate, data
        elif type == "image_caption" or type == 'ocr':
            return response_result[0]

        return response_result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example usage for TTS
text = ["Қазақ тілін болашақта қолданамын деп ойласыз ба?  Егер қолдансаңыз,  қай салаларда? Қолданбасаңыз, не себепті қолданбайсыз?"] # Қазақстанда қанша адам тұрады? қазақстанның астанасы қандай?
start_time = time.time()
result = get_embeddings(text, type="tts") # kazllm

wavfile.write("output.wav", result[0], result[1])
print(f"Total time is {time.time() - start_time}")

# Example usage for image captioning
with open("image8.jpg", "rb") as image_file:
    image_bytes = image_file.read()

start_time = time.time()
result = get_embeddings(image_bytes, type="ocr")
print(f"Caption: {result}")
print(f"Total time is {time.time() - start_time}")