FROM nvcr.io/nvidia/tritonserver:23.12-py3

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip cache purge

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && apt-get install -y wget && apt-get clean

RUN pip3 install llama-cpp-python
RUN pip3 install transformers==4.45.2
RUN pip3 install torch==2.4.1
RUN pip3 install torchvision==0.19.1
RUN pip3 install pillow==10.4.0
RUN pip3 install surya-ocr
RUN pip3 install scipy==1.10.1
RUN pip3 install vosk==0.3.45
RUN pip3 install onnxruntime-gpu==1.19.0
RUN pip3 install tensorrt==8.6.1
RUN pip3 install pycuda

# 0.8.3 for surya-ocr
# COPY download_models.sh .
# RUN chmod +x download_models.sh
# RUN ./download_models.sh

# COPY requirements.txt .
# RUN pip3 install -r requirements.txt

ENTRYPOINT [ "tritonserver" ]