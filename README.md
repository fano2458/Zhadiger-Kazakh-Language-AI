# AI Services for Kazakh Language

This repository provides AI services for the Kazakh language, including a language model for generating text completions.

## Models

Currently, it contains only one model:
- `kazllm`: A language model for Kazakh.

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
    │       └── checkpoints_llama8b_031224_18900-Q4_K_M.gguf
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
