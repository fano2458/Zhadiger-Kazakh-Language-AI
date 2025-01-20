import pickle
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from PIL import Image as PIL_Image
import base64
import numpy as np
import torch
import torchvision
import io
import triton_python_backend_utils as pb_utils


import sys
sys.path.append('/assets/image_caption')
from utils.language_utils import tokens2description, create_pad_mask, create_no_peak_and_pad_mask


class TritonPythonModel:
    def initialize(self, args):
        self.load_model()
        self.load_transforms()
        self.load_masks()

    def load_model(self):
        self.imgsz = [384, 384]
        self.device = torch.device('cuda:0')
        
        with open("/assets/image_caption/checkpoint/vocab_kz.pickle", 'rb') as f:
            self.coco_tokens = pickle.load(f)
            self.sos_idx = self.coco_tokens['word2idx_dict'][self.coco_tokens['sos_str']]
            self.eos_idx = self.coco_tokens['word2idx_dict'][self.coco_tokens['eos_str']]

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, '')  # initialize TensorRT plugins
        with open("/assets/image_caption/checkpoint/model_engine.trt", "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_tensor_shape(binding))
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def load_transforms(self):
        img_size = 384
        self.transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
        self.transf_2 = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_masks(self):
        batch_size = 1
        enc_exp_list = [32, 64, 128, 256, 512]
        dec_exp = 16
        num_heads = 8
        NUM_FEATURES = 144
        MAX_DECODE_STEPS = 20

        self.enc_mask = create_pad_mask(mask_size=(batch_size, sum(enc_exp_list), NUM_FEATURES),
                                    pad_row=[0], pad_column=[0]).contiguous()

        no_peak_mask = create_no_peak_and_pad_mask(mask_size=(batch_size, MAX_DECODE_STEPS, MAX_DECODE_STEPS),
                                                    num_pads=[0]).contiguous()

        cross_mask = create_pad_mask(mask_size=(batch_size, MAX_DECODE_STEPS, NUM_FEATURES),
                                    pad_row=[0], pad_column=[0]).contiguous()
        
        cross_mask = 1 - cross_mask

        self.fw_dec_mask = no_peak_mask.unsqueeze(2).expand(batch_size, MAX_DECODE_STEPS,
                                                        dec_exp, MAX_DECODE_STEPS).contiguous(). \
                            view(batch_size, MAX_DECODE_STEPS * dec_exp, MAX_DECODE_STEPS)

        self.bw_dec_mask = no_peak_mask.unsqueeze(-1).expand(batch_size,
                    MAX_DECODE_STEPS, MAX_DECODE_STEPS, dec_exp).contiguous(). \
                    view(batch_size, MAX_DECODE_STEPS, MAX_DECODE_STEPS * dec_exp)

        self.atten_mask = cross_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

    def preprocess_image(self, image_base64):
        image_bytes = base64.b64decode(image_base64)
        pil_image = PIL_Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = PIL_Image.new("RGB", pil_image.size)
        preprocess_pil_image = self.transf_1(pil_image)
        tens_image_1 = torchvision.transforms.ToTensor()(preprocess_pil_image)
        tens_image_2 = self.transf_2(tens_image_1)
        return tens_image_2.unsqueeze(0)

    def execute(self, requests):
        responses = []

        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "IMAGES").as_numpy()
            image_base64 = images[0].decode('utf-8')
            image = self.preprocess_image(image_base64)

            self.inputs[0]['host'] = np.ravel(image.numpy()).astype(np.float32)
            self.inputs[1]['host'] = np.array([0]).astype(np.int32)
            self.inputs[2]['host'] = np.array([self.sos_idx]).astype(np.int32)
            # transfer data to the gpu
            for inp in self.inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            # run inference
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle)
            # fetch outputs from gpu
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            # synchronize stream
            self.stream.synchronize()
            output_caption = tokens2description(self.outputs[0]['host'].tolist(), self.coco_tokens['idx2word_list'], self.sos_idx, self.eos_idx)
        
            output_tensor = pb_utils.Tensor("OUTPUT", np.array([output_caption], dtype=np.object_))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )

            responses.append(inference_response)

        return responses
