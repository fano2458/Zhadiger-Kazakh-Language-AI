import requests
import json
import base64
import time
import numpy as np
import io
from scipy.io import wavfile


def get_payload(data, type, role="", question=""):
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
    elif type == "translator":
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
    elif type == "stt":
        payload = {
            "inputs": [
                {
                    "name": "AUDIO",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [base64.b64encode(data).decode('utf-8')]
                }
            ]
        }
    elif type == "kazllm":
        payload = {
            "inputs": [
                {
                    "name": "TEXTS",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": data
                },
                {
                    "name": "TASK",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [role]
                },
                {
                    "name": "QUESTION",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [question]
                }
            ]
        }
    elif type == "tts" or type == "ner":
        payload = {
            "inputs": [
                {
                    "name": "TEXTS",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": data
                }
            ]
        }

    return payload


def get_response(data, type, role="", question=""):
    url = f"http://localhost:8500/v2/models/{type}/infer"
    headers = {
        'Content-Type': 'application/json',
    }

    payload = get_payload(data, type, role, question)

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        response_data = response.json()
        response_result = response_data['outputs'][0]['data']

        if type == "ner":
            response_result = json.loads(response_result[0])
        elif type == "tts":
            response_result = np.array(response_result, dtype=np.uint8)
            with io.BytesIO(response_result) as wav_io:
                rate, data = wavfile.read(wav_io)
                return rate, data
        
        return response_result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    

def test_tts():
    text = ["Қазақ тілін болашақта қолданамын деп ойласыз ба?  Егер қолдансаңыз,  қай салаларда? Қолданбасаңыз, не себепті қолданбайсыз?"] # Қазақстанда қанша адам тұрады? қазақстанның астанасы қандай?
    start_time = time.time()
    result = get_response(text, type="tts")
    wavfile.write("output.wav", result[0], result[1])
    print(f"Total time is {time.time() - start_time}")


def test_image_caption():
    with open("image8.jpg", "rb") as image_file:
        image_bytes = image_file.read()

    start_time = time.time()
    result = get_response(image_bytes, type="image_caption")
    print(f"Caption: {result}")
    print(f"Total time is {time.time() - start_time}")


def test_stt():
    with open("test.wav", "rb") as audio_file:
        audio_bytes = audio_file.read()

    start_time = time.time()
    result = get_response(audio_bytes, type="stt")
    print(f"Text: {result}")
    print(f"Total time is {time.time() - start_time}")


def test_ner():
    text = ["Қазақстан Республикасының астанасы - Астана."]
    start_time = time.time()
    result = get_response(text, type="ner")
    print(f"NER: {result}")
    print(f"Total time is {time.time() - start_time}")


def test_kazllm():
    text = ["Елдің елордасы — Астана қаласы. Мемлекеттік тілі — қазақ тілі. Орыс тілі мемлекеттік ұйымдарда және жергілікті өзін-өзі басқару органдарында "
        "ресми түрде қазақ тілімен тең қолданылады. Қазақстанның ұлттық құрамы алуан түрлі. Халықтың басым бөлігін тұрғылықты қазақ халқы құрайды, "
        "пайыздық үлесі — 70,18%,[9] орыстар — 18,42%, өзбектер — 3,29%, украиндар — 1,36%, ұйғырлар — 1,48%, татарлар — 1,06%, басқа халықтар 5,38%.[10] "
        "Халықтың 75% астамын мұсылмандар құрайды, православты христиандар — 21%, қалғаны басқа да дін өкілдері.[11] "
        "Экономикалық көрсеткіштері бойынша дамушы экономика ретінде қарастырылады. Елдің жалпы ішкі өнімі ЖІӨ (номинал) — $205,539 млрд (2018). "
        "Экономиканың негізгі бағыты — отын-энергетика саласындағы шикізат өндіру, ауыл шаруашылығы (егіншілік). Елдің негізгі валютасы — теңге. "]
    question = ["Қазақстанда православты христиандардың пайызы қанша?"]

    start_time = time.time()
    result = get_response(text, type="kazllm", role="qa", question=question)
    print(f"KazLLM: {result}")
    print(f"Total time is {time.time() - start_time}")


if __name__ == "__main__":
    test_tts()
    test_image_caption()
    test_stt()
    test_ner()
    test_kazllm()