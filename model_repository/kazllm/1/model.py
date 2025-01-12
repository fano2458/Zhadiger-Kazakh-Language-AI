from llama_cpp import Llama
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.llm = Llama(model_path="/assets/kazllm/checkpoints_llama8b_031224_18900-Q4_K_M.gguf")

    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "TEXTS").as_numpy()
            texts = [el.decode() for el in texts]

            prompt = [
                {
                    "role": "system",
                    "content": "Сіз көмекшісіз."
                },
                {
                    "role": "user",
                    "content": "Cәлеметсіз бе!"
                },
                {
                    "role": "assistant",
                    "content": "Сәлеметсіз бе! Vен сізге қалай көмектесе аламын?"
                },
                {
                    "role": "user",
                    "content": texts[0],
                },
            ]
            max_tokens = 500
            temperature = 0.75
            top_p = 0.1
            echo = True
            stop = ["Q", "\n"]

            generated_text = self.llm.create_chat_completion(prompt, temperature=temperature, top_p=top_p, stop=stop)
            response_content = generated_text["choices"][0]["message"]["content"]

            output_tensor = pb_utils.Tensor("OUTPUT", np.array(response_content, dtype=np.object_))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )

            responses.append(inference_response)

        return responses
