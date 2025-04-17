import requests
import json
import base64
import time
import numpy as np
import io
from scipy.io import wavfile


def get_payload(data, type, role="", question="", lang_type="", trg_lang="", file_paths=[]):
    if type == "image_caption" or type == 'ocr':
        encoded_data = base64.b64encode(data).decode('utf-8')
        payload = {
            "inputs": [
                {
                    "name": "texts",
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
                    "name": "texts",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": data
                },
                {
                    "name": "lang_type",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [lang_type]
                },
                {
                    "name": "trg_lang",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [trg_lang]
                }
            ]
        }
    elif type == "stt":
        payload = {
            "inputs": [
                {
                    "name": "audio",
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
                    "name": "texts",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": data
                },
                {
                    "name": "task",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [role]
                },
                {
                    "name": "question",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [question]
                }
            ]
        }
    elif type == "tts" or type == "ner" or type == "kazclip":
        payload = {
            "inputs": [
                {
                    "name": "texts",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": data
                }
            ]
        }
    elif type == "rag":
        payload = {
            "inputs": [
                {
                    "name": "texts",
                    "shape": [len(data)],
                    "datatype": "BYTES",
                    "data": data
                },
                {
                    "name": "user_request",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [question]
                },
                {
                    "name": "file_paths",
                    "shape": [len(file_paths)],
                    "datatype": "BYTES",
                    "data": file_paths
                }
            ]
        }

    return payload


def get_response(data, type, role="", question="", lang_type="", trg_lang="", file_paths=[]):
    url = f"http://localhost:8500/v2/models/{type}/infer"
    # url = f"https://shrew-above-absolutely.ngrok-free.app/v2/models/{type}/infer"
    headers = {
        'Content-Type': 'application/json',
    }

    payload = get_payload(data, type, role, question, lang_type, trg_lang, file_paths)

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # print(response.json())

    if response.status_code == 200:
        response_data = response.json()

        response_result = response_data['outputs'][0]['data']

        # print(response_result)

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
    with open("image1.jpg", "rb") as image_file:
        image_bytes = image_file.read()

    start_time = time.time()
    result = get_response(image_bytes, type="image_caption")
    print(f"Caption: {result}")
    print(f"Total time is {time.time() - start_time}")


def test_ocr():
    with open("image.png", "rb") as image_file:
        image_bytes = image_file.read()

    start_time = time.time()
    result = get_response(image_bytes, type="ocr")
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
    # text = ["Қазақстан Республикасының астанасы - Астана."]
    text = ["Сен қай жерде боласын?"]
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


def test_translator():
    text = ["Елдің елордасы — Астана қаласы."]
    start_time = time.time()
    result = get_response(text, type="translator", lang_type="kk", trg_lang="en")
    print(f"Translation: {result}")
    print(f"Total time is {time.time() - start_time}")


def test_kazclip():
    import base64
    from PIL import Image
    text = ["терезенің алдында тұрған адам"]
    start_time = time.time()
    result = get_response(text, type="kazclip")
    
    for i, image in enumerate(result):
        image = Image.open(io.BytesIO(base64.b64decode(image)))
        image.save(f"kazclip_{i}.png")

    print(f"KazClip: {len(result)}")
    print(f"Total time is {time.time() - start_time}")


def test_rag():
    texts = ["""Провост бойынша студенттерді оқуға қабылдауды қамтамасыз етуге жауапты
Университеттін кұрылымдық бөлімшесі:
3) Қабылдау кезеңі - бір немесе бірнеше қабылдау раундтарынан
тұратын, онлайн өтінімді, құжаттарды қарауды, бағалау мен қабылдау
процестерін қамтитын, Университетке өтініш берудің соңғы мерзімдерінің
жиынтығы:
4) Үміткер - іріктеу процесіне қатысатын Қазақстан Республикасының
азаматы, шетел азаматы немесе азаматтығы жоқ адам;
5) Сауалнама нысаны - Жеке кабинетте орналастырылған
магистратураға қабылдау түралы өтініш;
6) CGPA (Cumulative Grade Point Average) - студенттің дәреже алуына
ықпал ететін барлық оқу кезеңіндегі бүкіл бағаларының орташа мәнін есептеу;
7) Шартты түрде оқуға қабылдау - оқуға түсу кезеңінде
бағдарламалардың Қабылдау комиссиясының шешімімен және оқуға қабылдау
кезеңінде Университет Провостының немесе оны алмастыратын тұлғаның
шешімімен ресми белгіленген шартпен немесе шарттармен бағдарламаға
уміткерлерді кабылдау турі:
8) Оқуға қабылдауға келісу нысаны - Университет магистратурасына
қабылдау туралы хатты алу кезінде үміткерлер толтырған және қол қойған
нысан, ол оқуға түсү немесе қабылдаудан шығару үшін негіз болып табылады;
9) Сыртқы сарапшы - тиісті келісім бойынша Университеттің басқа
мектебінің немесе әріптес мекеменің немесе халықаралық танылған
университеттің немесе тиісті ресурстары (зияткерлік, ғылыми, әдістемелік, білім
беру, технологиялық, техникалық, адами) бар зерттеу институтының өз қызметін
келесі салалардың бірінде немесе бірнешеуінде жузеге асыратын: ғылыми-
зерттеу, заманауи әдіснама және білім беру технологиялары, білім беру
бағдарламаларын әзірлеу және іске асыру және т.б., үміткерлерге шолу және
бағалау жүргізуге қатысатын, сондай-ақ сапаны қамтамасыз ету үшін сыртқы
сарапшы немесе бағалаушылар ретінде әрекет ететін өкілі;
10) Назарбаев Университеті магистратура бағдарламаларының
дайындық жылы (НУМБДЖ) - ағылшын тілін жеделдетіп оқытуға және
Университет бағдарламаларына қабылданған, бірақ Мектептерге тікелей түсү
үшін қажетті деңгейде ағылшын тілін меңгермеген студенттер үшін оқыған
материалдарын қайталауға бағытталған несие берілмейтін екі семестрлік
күндізгі бағдарлама;
11) Жеке кабинет - Университет сайтында тіркелу кезінде жеке кеңістік
ұсынатын онлайн сервис (www.admissions.nu.edu.kz);
12) Бағдарлама - Медицина мектебінің «Медицина докторы»
бағдарламасынан және Жоғары бизнес мектебі Басшыларға арналған іскерлік
әкімшілендіру магистрі және Адами ресурстарды басқару жөніндегі магистр
бағдарламаларынан басқа, Мектептер ұсынатын магистратура бағдарламалары;
13) Мектеп өкілі - Университеттегі академиялық және әкімшілік
қызметтерді басқаруға қатысатын оқытушылар, әкімшілік немесе Мектеп
басшылығы:""",
""""
Провосттың
2021 жылғы «15» қазандағы
№ 132-н/к шешіміне қосымша
«Назарбаев Университеті»
дербес білім беру ұйымы
Провост
Бекіткен орган:
Назарбаев Университеті дербес білім беру ұйымының магистратура
бағдарламаларына қабылдау қағидалары
Купніне
15.10.2021
15.10.2021
Бекітілген күні:
енгізілген күні
Шешімнін/хаттаманын
№ 132-н/к
№:
ІНҚ жіктеуші:
2.1.2. Master's degree
А. Жазыкпаева Студенттерді оқуға қабылдау
Бастамашы:
департаментінің директоры
Өзара байланысты
Өзара байланысты құжаттар жоқ
кужаттар
1. Жалпы ережелер
Осы «Назарбаев Университеті» дербес білім беру ұйымының
1.
магистратура бағдарламаларына қабылдау қағидалары (бұдан әрі - Кағилалар):
1) «Назарбаев Университеті», «Назарбаев Зияткерлік мектептері» және
«Назарбаев Қоры» мәртебесі туралы» Қазақстан Республикасының 2011 жылғы
19 қаңтардағы Заңына;
2) Жоғары қамқоршылық кеңестің 2013 жылғы 18 сәуірдегі қаулысымен
бекітілген «Назарбаев Университеті» дербес білім беру ұйымының Жарғысына;
3) Жоғары қамқоршылық кеңестің 2018 жылғы 1 желтоқсандағы
қаулысымен бекітілген Назарбаев Университетінің 2018-2030 жылдарға
арналған стратегиясының негізгі бағыттарына сәйкес әзірленді және «Назарбаев
Университеті» дербес білім беру ұйымының магистратура бағдарламаларына
кабылдау тәртібін айкындайды.
2. Осы Қағидаларда пайдаланылатын негізгі ұғымдар мен қысқартулар:
l ) Қабылдау комиссиясы - магистратура бағдарламаларына қабылдау
рәсімдерін іске асыру үшін құрылған Университеттің консультативтік органы;
Студенттерді оқуға қабылдау департаменті - дайындық
2) 
бағдарламалары, бакалавриат, магистратура және докторантура бағдарламалары

"""]
    user_request = ["Нұрсұлтан Назарбаев неше жаста?"]
    file_paths = ["1.txt","2.txt"]

    result = get_response(texts, type="rag", question=user_request, file_paths=file_paths)
    # print(result)
    for r in result:
        print(r)
        print("----")

if __name__ == "__main__":
    # print("tts time: ")
    # test_tts()
    # print("image caption time: ")
    # test_image_caption()
    # test_stt()
    # print("ner time: ")
    # test_ner()
    # # test_kazllm()
    # print("ocr time: ")
    # test_ocr()
    # print("translator time: ")
    test_translator()
    # print("kazclip time: ")
    # test_kazclip()
    # test_rag()
