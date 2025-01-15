#!/bin/bash

# Model weights
IMAGE_CAPTION_WEIGHTS="assets/image_caption/checkpoint/kaz_model.pth"
KAZLLM_WEIGHTS="assets/kazllm/checkpoint/checkpoints_llama8b_031224_18900-Q4_K_M.gguf"
NER_WEIGHTS="assets/ner/checkpoint/model.safetensors"
TRANSLATOR_WEIGHTS="/assets/translator/checkpoint/pytorch_model.bin"
TTS_WEIGHTS="assets/tts/checkpoint/model.safetensors"

# URLs for downloading weights
IMAGE_CAPTION_URL="	https://drive.usercontent.google.com/download?id=15C0NBneOhfSC5SYn01tL5E5flSM4CXB7&export=download&authuser=0&confirm=t&uuid=64f53d75-dfd7-49ab-b0b3-f0d8508b1e72&at=AIrpjvPPDfZM-h8SayfrpWszrtvu:1736968124335"
KAZLLM_URL="https://drive.usercontent.google.com/download?id=1nsl2f_rh7OgPRXwTEb2RJNYeE4xWA0CR&export=download&authuser=0&confirm=t&uuid=3dd15dbb-c521-488f-b168-d98feb2c28cd&at=AIrpjvNFYnZarcD7Km0fsepqmrq8:1736968097448"
NER_URL="https://drive.usercontent.google.com/download?id=1ih9t7UmfqE4DbUGOYhn8wCvgJ2fZni8r&export=download&authuser=0&confirm=t&uuid=75a22c7c-a90b-42e5-8ab5-2b079cb803cc&at=AIrpjvOyEsiJi5EW6V88t1m7_LAs:1736968060272"
TRANSLATOR_URL="https://drive.usercontent.google.com/download?id=10i99MD5JgcA1zStJ_YBIN67f09xB5iV1&export=download&authuser=0&confirm=t&uuid=01ab9d34-2be2-478c-a066-bebb32d32f3a&at=AIrpjvNSy3ij_70Ttq5ehp_pCpKh:1736968012669"
TTS_URL="https://drive.usercontent.google.com/download?id=1EZr7uXfbUYVEJnjcOFVYtul80KTPRgeu&export=download&authuser=0&confirm=t&uuid=0014e972-ba2f-4659-8755-221558bc000d&at=AIrpjvMES-hK9YUrD91PJKikYyzf:1736967893962"

# Function to download weights
download_if_not_exists() {
    local file_path=$1
    local url=$2

    if [ ! -f "$file_path" ]; then
        echo "Downloading $file_path..."
        wget -O "$file_path" "$url"
    else
        echo "$file_path already exists."
    fi
}

# Download weights
download_if_not_exists "$IMAGE_CAPTION_WEIGHTS" "$IMAGE_CAPTION_URL"
download_if_not_exists "$KAZLLM_WEIGHTS" "$KAZLLM_URL"
download_if_not_exists "$NER_WEIGHTS" "$NER_URL"
download_if_not_exists "$TRANSLATOR_WEIGHTS" "$TRANSLATOR_URL"
download_if_not_exists "$TTS_WEIGHTS" "$TTS_URL"
