import sys
from basicsr.utils import imwrite
sys.path.insert(0, '../src')

from gfpgan_server import Inference

if __name__ == '__main__':
    i = Inference('1.4')

    _, _, img = i('00.jpg')
    imwrite(img, '00.result.jpg')
