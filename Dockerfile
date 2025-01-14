FROM nvcr.io/nvidia/tritonserver:23.12-py3
# FROM kaz_ai_triton-triton_ai_services:latest

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "tritonserver" ]