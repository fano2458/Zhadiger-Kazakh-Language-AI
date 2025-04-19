from llama_cpp import Llama
import triton_python_backend_utils as pb_utils
import numpy as np
from threading import Thread
import re


class TritonPythonModel:
    def initialize(self, args):
        self.llm = Llama(model_path="/assets/kazllm/checkpoint/checkpoints_llama8b_031224_18900-Q4_K_M.gguf", n_ctx = 2048*4, flash_attn=True, n_gpu_layers=0)
        
        self.language_prompts = {
            "kk": {
                "system_role": {
                    "summarization": "мәтінді қысқаратын",
                    "qa": "сұраққа жауап беретін",
                    "chat": "жалпы мақсаттағы сөйлесу көмекшісі"
                },
                "greeting_user": "Cәлеметсіз бе!",
                "greeting_assistant": "Сәлеметсіз бе! Мен сізге қалай көмектесе аламын?",
                "instruction": {
                    "summarization": "Келесі мәтіннің қысқаша мазмұнын беріңіз\n",
                    "qa": "Келесі мәтін мазмұнын бойынша келесі сұраққа жауап беріңізегер, ақпарат мәтінде болмаса, білмеймін деп айтыңыз\n",
                    "chat": ""
                }
            },
            "ru": {
                "system_role": {
                    "summarization": "помощник по сокращению текста",
                    "qa": "помощник по ответам на вопросы",
                    "chat": "ассистент общего назначения"
                },
                "greeting_user": "Здравствуйте!",
                "greeting_assistant": "Здравствуйте! Чем я могу вам помочь?",
                "instruction": {
                    "summarization": "Предоставьте краткое содержание следующего текста\n",
                    "qa": "Ответьте на следующий вопрос на основе содержания текста. Если информации нет в тексте, скажите, что не знаете\n",
                    "chat": ""
                }
            },
            "en": {
                "system_role": {
                    "summarization": "text summarization assistant",
                    "qa": "question answering assistant",
                    "chat": "general purpose conversational assistant"
                },
                "greeting_user": "Hello!",
                "greeting_assistant": "Hello! How can I help you?",
                "instruction": {
                    "summarization": "Please provide a summary of the following text\n",
                    "qa": "Please answer the following question based on the content of the text. If the information is not in the text, say that you don't know\n",
                    "chat": ""
                }
            }
        }
        
        self.default_language = "kk"

    def detect_language(self, text):
        """Simple language detection based on character patterns"""
        if re.search(r'[әіңғүұқөһ]', text.lower()):
            return "kk"
        
        if re.search(r'[а-яА-Я]', text):
            return "ru"
            
        return "en"

    def execute(self, requests):
        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            texts = [el.decode() for el in texts]

            formatted_texts = ""
            for i, text in enumerate(texts):
                formatted_texts += text

            task = pb_utils.get_input_tensor_by_name(request, "task").as_numpy()
            task = [el.decode() for el in task][0]

            # Check if task is empty, if so, set to chat mode
            if not task.strip():
                task = "chat"
                # In chat mode, the message is in the texts field
                user_message = formatted_texts
                question = ""
            else:
                question = pb_utils.get_input_tensor_by_name(request, "question").as_numpy()
                question = [el.decode() for el in question][0]
            
            detect_from = question if task != "chat" else formatted_texts
            detected_lang = self.detect_language(detect_from)
            
            lang_prompts = self.language_prompts.get(detected_lang, self.language_prompts[self.default_language])

            role = lang_prompts["system_role"][task]
            instruction = lang_prompts["instruction"][task]

            if task == "chat":
                user_content = user_message
            else:
                user_content = instruction + formatted_texts + '\n' + question

            prompt = [
                {
                    "role": "system",
                    "content": f"Вы {role}." if detected_lang == "ru" else 
                               f"You are a {role}." if detected_lang == "en" else
                               f"Сіз {role} көмекшісіз."
                },
                {
                    "role": "user",
                    "content": lang_prompts["greeting_user"]
                },
                {
                    "role": "assistant",
                    "content": lang_prompts["greeting_assistant"]
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
            max_tokens = 512
            temperature = 0.2
            top_p = 0.1

            generation_kwargs = {
                "messages": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True
            }

            response_sender = request.get_response_sender()

            def run_inference():
                for chunk in self.llm.create_chat_completion(**generation_kwargs):
                    delta = chunk["choices"][0]["delta"]
                    if 'content' in delta:
                        partial_text = delta['content']
                        out_output = pb_utils.Tensor(
                            "output", np.array([partial_text], dtype=np.object_)
                        )
                        response_sender.send(
                            pb_utils.InferenceResponse(output_tensors=[out_output])
                        )
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

        return None
