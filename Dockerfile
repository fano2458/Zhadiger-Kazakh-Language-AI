FROM nvcr.io/nvidia/tritonserver:23.12-py3
# FROM kaz_ai_triton-triton_ai_services:latest

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip cache purge

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && apt-get install -y wget && apt-get clean

RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" FORCE_CMAKE=1 pip3 install llama-cpp-python==0.3.6
# RUN pip3 install llama-cpp-python==0.3.6
RUN pip3 install transformers==4.45.2
RUN pip3 install torch==2.4.1
RUN pip3 install torchvision==0.19.1
RUN pip3 install pillow==10.4.0
RUN pip3 install surya-ocr==0.8.3
RUN pip3 install scipy==1.10.1
RUN pip3 install vosk==0.3.45
RUN pip3 install onnxruntime-gpu==1.19.0
RUN pip3 install tensorrt==8.6.1
RUN pip3 install pycuda==2024.1.2

# COPY download_models.sh .
# RUN chmod +x download_models.sh
# RUN ./download_models.sh

# COPY requirements.txt .
# RUN pip3 install -r requirements.txt

# COPY onnx2trt.py .
# RUN python3 onnx2trt.py

ENTRYPOINT [ "tritonserver" ]