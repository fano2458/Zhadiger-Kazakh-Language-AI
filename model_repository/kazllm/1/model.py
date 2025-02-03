from llama_cpp import Llama
import triton_python_backend_utils as pb_utils
import numpy as np
from threading import Thread


class TritonPythonModel:
    def initialize(self, args):
        self.llm = Llama(model_path="/assets/kazllm/checkpoint/checkpoints_llama8b_031224_18900-Q4_K_M.gguf", n_ctx = 2048, flash_attn=True, n_gpu_layers=-1)

    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            texts = [el.decode() for el in texts][0]

            task = pb_utils.get_input_tensor_by_name(request, "task").as_numpy()
            task = [el.decode() for el in task][0]

            question = pb_utils.get_input_tensor_by_name(request, "question").as_numpy()
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

            generation_kwargs = {
                "messages": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True
            }

            response_sender = request.get_response_sender()
            # full_text_chunks = []

            def run_inference():
                for chunk in self.llm.create_chat_completion(**generation_kwargs):
                    delta = chunk["choices"][0]["delta"]
                    if 'content' in delta:
                        partial_text = delta['content']
                        # full_text_chunks.append(partial_text)
                        out_output = pb_utils.Tensor(
                            "output", np.array([partial_text], dtype=np.object_)
                        )
                        response_sender.send(
                            pb_utils.InferenceResponse(output_tensors=[out_output])
                        )
                # final_text = "".join(full_text_chunks)
                final_text = "\n"
                output_tensor = pb_utils.Tensor(
                    "output", np.array(final_text, dtype=np.object_)
                )
                final_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                response_sender.send(
                    final_response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

            thread = Thread(target=run_inference)
            thread.start()
            thread.join()

        # return responses
        return None
