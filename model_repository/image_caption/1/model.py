import pickle
import torch
import numpy as np
from argparse import Namespace
import triton_python_backend_utils as pb_utils
from PIL import Image as PIL_Image

import sys
sys.path.append('/assets/image_caption')
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.language_utils import convert_vector_idx2word


class TritonPythonModel:
    def initialize(self, args):
        load_path = "/assets/image_caption/checkpoint/kaz_model.pth"
        dict_path = "/assets/image_caption/checkpoint/vocab_kz.pickle"

        drop_args = Namespace(enc=0.0,
                              dec=0.0,
                              enc_input=0.0,
                              dec_input=0.0,
                              other=0.0)
        
        model_args = Namespace(model_dim=512,
                               N_enc=3,
                               N_dec=3,
                               dropout=0.0,
                               drop_args=drop_args)
        
        with open(dict_path, "rb") as f:
            self.coco_tokens = pickle.load(f)

        img_size = 384
        self.model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                        swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                        swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                        swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                        swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                        swin_use_checkpoint=False,
                                        final_swin_dim=1536,

                                        d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                        N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                        num_exp_enc_list=[32, 64, 128, 256, 512],
                                        num_exp_dec=16,
                                        output_word2idx=self.coco_tokens['word2idx_dict'],
                                        output_idx2word=self.coco_tokens['idx2word_list'],
                                        max_seq_len=63, drop_args=model_args.drop_args,
                                        rank=0)
        
        device = torch.device("cpu")
        self.model.to(device)

        checkpoint = torch.load(load_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.transf_1 = lambda img: img.resize((img_size, img_size), PIL_Image.BICUBIC)
        self.transf_2 = lambda img: (np.array(img) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        self.model.eval()

        """### Beam search configuration"""

        self.beam_search_kwargs = {'beam_size': 5,
                            'beam_max_seq_len': 63,
                            'sample_or_max': 'max',
                            'how_many_outputs': 1,
                            'sos_idx': self.coco_tokens['word2idx_dict'][self.coco_tokens['sos_str']],
                            'eos_idx': self.coco_tokens['word2idx_dict'][self.coco_tokens['eos_str']]}
        
    def execute(self, requests):
        responses = []

        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "IMAGES").as_numpy()
            images = [el.decode() for el in images]
            print(images)
            pil_image = PIL_Image.open(images[0])
            if pil_image.mode != 'RGB':
                pil_image = PIL_Image.new("RGB", pil_image.size)
            preprocess_pil_image = self.transf_1(pil_image)
            tens_image_1 = np.transpose(np.array(preprocess_pil_image), (2, 0, 1))
            tens_image_2 = self.transf_2(tens_image_1)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image = torch.tensor(tens_image_2, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred, _ = self.model(enc_x=image,
                                enc_x_num_pads=[0],
                                mode='beam_search', **self.beam_search_kwargs)
            pred = convert_vector_idx2word(pred[0][0], self.coco_tokens['idx2word_list'])[1:-1]
            pred[-1] = pred[-1] + '.'
            pred = ' '.join(pred).capitalize()

            output_tensor = pb_utils.Tensor("OUTPUT", np.array([pred], dtype=np.object_))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )

            responses.append(inference_response)

        return responses

