#!/bin/bash

# Model weights
IMAGE_CAPTION_WEIGHTS="/assets/image_caption/checkpoint/kaz_model.pth"
KAZLLM_WEIGHTS="/assets/kazllm/checkpoint/checkpoints_llama8b_031224_18900-Q4_K_M.gguf"
NER_WEIGHTS="/assets/ner/checkpoint/model.safetensors"
TRANSLATOR_WEIGHTS="/assets/translator/checkpoint/pytorch_model.bin"
TTS_WEIGHTS="/assets/tts/checkpoint/model.safetensors"

# URLs for downloading weights
IMAGE_CAPTION_URL=""
KAZLLM_URL=""
NER_URL=""
TRANSLATOR_URL=""
TTS_URL=""

# Function to download weights
download_if_not_exists() {
    local file_path=$1
    local url=$2

    if [ ! -f "$file_path"]; then
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
