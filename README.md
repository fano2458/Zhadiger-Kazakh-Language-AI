# AI Services for Kazakh Language

This repository provides AI services for the Kazakh language, including a language model for generating text completions.

## Models

Currently, it contains 7 models:
- `kazllm`: A language model for Kazakh, capable of chatting, answering questions, and text summarization.
- `ner`: Model for named enteties recognition.
- `translator`: Model for translation from Kazakh to English and vice versa.
- `tts`: Model that generates speech from text written on Kazakh.
- `ocr`: Model that does optical character recognition on an image.
- `image_caption`: Model that generates short description for a given image.
- `stt`: Model that generates text from a given speech recording.

## TODO List

- [ ] Add image generator (text to image).
- [ ] Add vqa model (visual question answering).
- [ ] Convert current models to ONNX format.
- [ ] Convert ONNX models to TensorRT engine.
- [ ] Warm up for some of the models.
- [ ] Response Cache for some of the models.
- [ ] Measure performance by performance analyzer (try out concurrent model execution) [link](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_2-improving_resource_utilization)
- [ ] Customization of deployment with Model analyzer [link](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_3-optimizing_triton_configuration)
- [ ] Accelerate model inference [link](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_4-inference_acceleration)
- [ ] Create pipelines with ensembles [link](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_5-Model_Ensembles) and BLS [link](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_6-building_complex_pipelines)

## Getting Started

### Prerequisites

- Docker
- NVIDIA Container Toolkit

### Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/fano2458/Zhadiger-Kazakh-Language-AI.git
    cd Zhadiger-Kazakh-Language-AI
    ```

2. Download the models:
    ```sh
    chmod +x download_models.sh
    ./download_models.sh    
    ```

3. Build and start the Docker containers:
    ```sh
    docker-compose up --build
    ```

### Testing the Model

You can test the models by using `request_test.py` script:

```sh
python request_test.py
```

LLM streaming output can be tested using `test_streaming.py` script:

```sh
python test_streaming.py
```

### Repository Structure

- `model_repository/`: Contains the model definition and configuration for Triton Inference Server.
- `assets/`: Directory where model weights should be placed.
- `request_test.py`: Script to test the model by sending a request and printing the response.

<!-- ### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- Special thanks to the contributors and the open-source community for their valuable work and support. -->
