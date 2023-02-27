import os
import cv2
import requests
from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from .utils import GFPGANer

data_root = os.path.join(os.path.dirname(__file__), 'data')


class Inference:
    """ GFPGAN

        Args:
            ver (str): GFPGAN model version. Option: 1.4 | RestoreFormer. Default: 1.4 
            upscale (int): The final upsampling scale of the image. Default: 2
            bg_tile (int, optional): Tile size for background sampler, 0 for no tile during testing. Defaults to 400.
            gpu_id (int, optional): deploy the model to gpu_idx. Defaults to 0.

        Raises:
            ValueError: _description_
    """
    def __init__(self, ver:str, upscale:int=2, bg_tile:int=400, gpu_id=0):
        
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.bg_upsampler = RealESRGANer(
            scale=2,
            model_path=os.path.join(data_root, 'RealESRGAN_x2plus.pth'),
            model=model,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=True, # need to set False in CPU mode
            gpu_id=gpu_id)
        
        if ver == '1.4':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
        elif ver == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
        else:
            raise ValueError(f'Wrong model version {ver}.')
        
        self.restorer = GFPGANer(
            model_path=os.path.join(data_root, model_name + '.pth'),
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=self.bg_upsampler,
            device='cuda:%d' % gpu_id)

    @staticmethod
    def __read_imocv(inp_buf):
        if isinstance(inp_buf, bytes):
            img = np.asarray(bytearray(inp_buf), dtype='uint8')
            image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        elif isinstance(inp_buf, str) and os.path.isfile(inp_buf):
            image = cv2.imread(inp_buf, cv2.IMREAD_COLOR)
        elif isinstance(inp_buf, str) and 'http' == inp_buf.strip()[:4]:
            resp = requests.get(inp_buf)
            image = cv2.imdecode(np.fromstring(resp.content, np.unit8), 1)
        elif isinstance(inp_buf, np.ndarray):
            image = inp_buf
        elif isinstance(inp_buf, Image.Image):
            image = cv2.cvtColor(np.asarray(inp_buf), cv2.COLOR_RGB2BGR)
        else:
            image = None
            raise ValueError('Error: Not support this type image buffer(byte, str, np.ndarry, PIL.Image)')

        return image

    def __call__(self, img_buf, weight=0.5, aligned=False, only_center_face=False):
        input_image = self.__read_imocv(img_buf) 
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            input_image,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=True,
            weight=weight)
        return cropped_faces, restored_faces, restored_img