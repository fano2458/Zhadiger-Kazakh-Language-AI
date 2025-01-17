import requests
import json
import time
import io
from scipy.io import wavfile
import numpy as np
import base64


def get_embeddings(data, type="ner", role="", question=""):
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
        elif type == "image_caption" or type == 'ocr' or type == 'stt':
            print(response_data)
            return response_result[0]

        return response_result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example usage for TTS
# text = ["Қазақ тілін болашақта қолданамын деп ойласыз ба?  Егер қолдансаңыз,  қай салаларда? Қолданбасаңыз, не себепті қолданбайсыз?"] # Қазақстанда қанша адам тұрады? қазақстанның астанасы қандай?
# start_time = time.time()
# result = get_embeddings(text, type="tts") # kazllm

# wavfile.write("output.wav", result[0], result[1])
# print(f"Total time is {time.time() - start_time}")

# # Example usage for image captioning
# with open("image8.jpg", "rb") as image_file:
#     image_bytes = image_file.read()

# start_time = time.time()
# result = get_embeddings(image_bytes, type="image_caption")
# print(f"Caption: {result}")
# print(f"Total time is {time.time() - start_time}")


# Example usage for Speech to Text
# with open("test.wav", "rb") as audio_file:
#     audio_bytes = audio_file.read()

# start_time = time.time()
# result = get_embeddings(audio_bytes, type="stt")
# print(f"Text: {result}")
# print(f"Total time is {time.time() - start_time}")

# Example usage for text summarization 
text = ["Қазақстан Республикасы (Дыбысы Қазақстан Республикасы) — Шығыс Еуропа мен Орталық Азияда орналасқан мемлекет. "
        "Батысында Еділдің төменгі ағысынан, шығысында Алтай тауларына дейін 3 000 километрге, солтүстіктегі Батыс Сібір жазығынан, оңтүстіктегі "
        "Қызылқұм шөлі мен Тянь-Шань тау жүйесіне 1 600 километрге созылып жатыр. Қазақстан Каспий көлі арқылы Әзербайжан, Иран елдеріне, "
        "Еділ өзені және Еділ-Дон каналы арқылы Азов теңізі мен Қара теңізге шыға алады. Мұхитқа тікелей шыға алмайтын мемлекеттердің ішінде Қазақстан — ең үлкені."
        "Қазақстан бес мемлекетпен шекаралас, соның ішінде әлемдегі құрлықтағы ең ұзын шекара, солтүстігінде және батысында Ресеймен — 7 591 км құрайды. "
        "Оңтүстігінде: Түрікменстан — 426 км, Өзбекстан — 2 354 км және Қырғызстан — 1 241 км, ал шығысында: Қытаймен — 1 782 км шектеседі. "
        "Жалпы құрлық шекарасының ұзындығы — 13 394 километр. Батыста Каспий көлімен (2000 км), оңтүстік батыста Арал теңізімен шайылады.[8] "
        "2024 жылғы 1 наурыздағы елдегі тұрғындар саны — 20 075 271,[3] бұл әлем бойынша 64-орын. Жер көлемі жағынан әлем елдерінің ішінде 9-орын алады (2 724 902 км²)"]

text = ["Елдің елордасы — Астана қаласы. Мемлекеттік тілі — қазақ тілі. Орыс тілі мемлекеттік ұйымдарда және жергілікті өзін-өзі басқару органдарында "
        "ресми түрде қазақ тілімен тең қолданылады. Қазақстанның ұлттық құрамы алуан түрлі. Халықтың басым бөлігін тұрғылықты қазақ халқы құрайды, "
        "пайыздық үлесі — 70,18%,[9] орыстар — 18,42%, өзбектер — 3,29%, украиндар — 1,36%, ұйғырлар — 1,48%, татарлар — 1,06%, басқа халықтар 5,38%.[10] "
        "Халықтың 75% астамын мұсылмандар құрайды, православты христиандар — 21%, қалғаны басқа да дін өкілдері.[11] "
        "Экономикалық көрсеткіштері бойынша дамушы экономика ретінде қарастырылады. Елдің жалпы ішкі өнімі ЖІӨ (номинал) — $205,539 млрд (2018). "
        "Экономиканың негізгі бағыты — отын-энергетика саласындағы шикізат өндіру, ауыл шаруашылығы (егіншілік). Елдің негізгі валютасы — теңге. "]

question = ["Қазақстанда православты христиандардың пайызы қанша?"]

start_time = time.time()
result = get_embeddings(question, type="kazllm")
print(f"Summary: {result}")
print(f"Total time is {time.time() - start_time}")