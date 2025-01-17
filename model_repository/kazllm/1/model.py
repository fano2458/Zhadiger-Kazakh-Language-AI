from llama_cpp import Llama
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.llm = Llama(model_path="/assets/kazllm/checkpoint/checkpoints_llama8b_031224_18900-Q4_K_M.gguf", n_ctx = 2048, flash_attn=True) # , n_gpu_layers=-1

    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "TEXTS").as_numpy()
            texts = [el.decode() for el in texts][0]

            task = pb_utils.get_input_tensor_by_name(request, "TASK").as_numpy()
            task = [el.decode() for el in task][0]

            question = pb_utils.get_input_tensor_by_name(request, "QUESTION").as_numpy()
            question = [el.decode() for el in question][0]

            instruction = ""
            role = ""

            if task == "summarization":
                instruction = "Келесі мәтіннің қысқаша мазмұнын беріңіз\n"
                role = "мәтінді қысқаратын"
            elif task == "qa":
                instruction = "Келесі мәтін мазмұнын бойынша келесі сұраққа жауап беріңіз\n"
                role = "сұраққа жауап беретін"

            prompt = [
                {
                    "role": "system",
                    "content": f"Сіз {role} көмекшісіз."
                },
                {
                    "role": "user",
                    "content": "Cәлеметсіз бе!"
                },
                {
                    "role": "assistant",
                    "content": "Сәлеметсіз бе! Мен сізге қалай көмектесе аламын?"
                },
                {
                    "role": "user",
                    "content": instruction + texts + '\n' + question,
                },
            ]
            max_tokens = 2048
            temperature = 0.75
            top_p = 0.1
            echo = True
            stop = ["Q", "\n"]

            generated_text = self.llm.create_chat_completion(prompt, temperature=temperature, top_p=top_p, stop=stop, max_tokens=max_tokens)
            response_content = generated_text["choices"][0]["message"]["content"]

            output_tensor = pb_utils.Tensor("OUTPUT", np.array(response_content, dtype=np.object_))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )

            responses.append(inference_response)

        return responses
