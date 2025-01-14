# AI Services for Kazakh Language

This repository provides AI services for the Kazakh language, including a language model for generating text completions.

## Models

Currently, it contains four models:
- `kazllm`: A language model for Kazakh.
- `ner`: Model for named enteties recognition.
- `image_caption`: Model for caption generation for images.
- `translator`: Model for translation from one Kazakh to English and vice versa.

## TODO List

- [ ] Check image_caption model (request_test.py).
- [ ] Check ocr, tts, stt.
- [ ] Convert current models to ONNX format.
- [ ] Convert ONNX models to TensorRT engine.
- [ ] Add more models to the repository.

## Getting Started

### Prerequisites

- Docker

### Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/fano2458/kaz_ai_triton.git
    cd kaz_ai_triton
    ```

2. Download the weights of the models and place them into the `assets` folder. Ensure the folder structure is as follows:
    ```
    kaz_ai_triton/
    ├── assets/
    │   └── kazllm/
    │       └── checkpoint/
    |           └── model_name.gguf
    ├── model_repository/
    ├── request_test.py
    ├── docker-compose.yaml
    └── ...
    ```

3. Build and start the Docker containers:
    ```sh
    docker-compose up --build
    ```

### Testing the Model

You can test the model using the `request_test.py` script:

```sh
python request_test.py
```

This script sends a request to the model server and prints the generated output for the input text.

### Repository Structure

- `model_repository/`: Contains the model definition and configuration for Triton Inference Server.
- `assets/`: Directory where model weights should be placed.
- `request_test.py`: Script to test the model by sending a request and printing the response.

<!-- ### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- Special thanks to the contributors and the open-source community for their valuable work and support. -->
