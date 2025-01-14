import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model
from scipy.io.wavfile import write
from pathlib import Path
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):

        self.fs = 22050

        # Load vocoder
        vocoder_checkpoint = "/home/fano/Downloads/parallelwavegan_male1_checkpoint/checkpoint-400000steps.pkl"
        self.vocoder = load_model(vocoder_checkpoint).to("cuda").eval()
        self.vocoder.remove_weight_norm()

        # Load Tacotron2 model
        config_file = "/home/fano/Downloads/kaztts_male1_tacotron2_train.loss.ave/exp/tts_train_raw_char/config.yaml"
        model_path = "/home/fano/Downloads/kaztts_male1_tacotron2_train.loss.ave/exp/tts_train_raw_char/train.loss.ave_5best.pth"

        self.text2speech = Text2Speech(
            config_file,
            model_path,
            device="cuda",
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=True,
            backward_window=1,
            forward_window=3,
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            text_input = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            text_input = text_input[0].decode("utf-8")

            with torch.no_grad():
                output_dict = self.text2speech(text_input.lower())
                feat_gen = output_dict['feat_gen']

                wav = self.vocoder.inference(feat_gen)

            wav_data = wav.view(-1).cpu().numpy()

            output_tensor = pb_utils.Tensor(
                "OUTPUT_AUDIO",
                np.array([wav_data], dtype=object)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )

            responses.append(inference_response)

        return responses
