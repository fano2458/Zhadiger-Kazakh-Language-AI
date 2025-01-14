import requests
import json
import time


def get_embeddings(text, type="ner"):
    url = f"http://localhost:8500/v2/models/{type}/infer"
    
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
            },
            {
                "name": "LANG_TYPE",
                "shape": [1],
                "datatype": "BYTES",
                "data": ["kaz"]
            },
            {
                "name": "TRG_LANG",
                "shape": [1],
                "datatype": "BYTES",
                "data": ["eng_Latn"]
            }
        ]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        response_data = response.json()
        embeddings = response_data['outputs'][0]['data']

        if type == "ner":
            embeddings = json.loads(embeddings[0])

        return embeddings
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


text = ["Қазақ тілін болашақта қолданамын деп ойласыз ба?  Егер қолдансаңыз,  қай салаларда? Қолданбасаңыз, не себепті қолданбайсыз?"] # Қазақстанда қанша адам тұрады? қазақстанның астанасы қандай?
start_time = time.time()
embeddings = get_embeddings(text, type="translator") # kazllm
print(embeddings)
print(f"Total time is {time.time() - start_time}")
# print(len(embeddings))