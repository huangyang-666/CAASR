import sys, os, random
import cv2, torch
from multiprocessing import Process, Queue
import numpy as np

root_path = os.path.abspath('.')
sys.path.append(root_path)
# Import files from the local folder
from opt import opt
from degradation.ESR.utils import tensor2np, np2tensor

class JPEG:
    def __init__(self) -> None:
        # Choose an image compression degradation
        pass

    #通过降低图像的颜色深度来模拟色带效果，首先将颜色值向下映射到较少的颜色级别，然后再映射回远来的范围，这样会在图像中产生明显色带
    #todo 通过level值控制色带强度，需要调试
    def simulate_banding(self, image, levels=10):
        """ Simulate color banding effect by reducing color levels and then restoring """
        banding_image = np.floor(image / levels) * (levels)
        return banding_image.astype(np.uint8)

    def compress_and_store(self, np_frames, store_path, idx):
        ''' Compress and Store the whole batch as JPEG
        Args:
            np_frames (numpy):      The numpy format of the data (Shape:?)
            store_path (str):       The store path
        Return:
            None
        '''

        # Preparation
        single_frame = np_frames

        # Simulate banding before compression
        single_frame = self.simulate_banding(single_frame, levels=20)
        # single_frame = self.simulate_banding(single_frame, levels=10)

        # Compress as JPEG
        jpeg_quality = random.randint(*opt['jpeg_quality_range2'])

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode('.jpg', single_frame, encode_param)
        decimg = cv2.imdecode(encimg, 1)

        # # Simulate banding after decompression to exaggerate the effect
        # decimg = self.simulate_banding(decimg, levels=30)   #10-30

        # Store the image with quality
        cv2.imwrite(store_path, decimg)



if __name__ == '__main__':
    # 读取输入图片
    input_image_path = r"E:\AnimeSR\compare_5\banding\bj_0_6.png"
    input_image = cv2.imread(input_image_path)

    # 创建 JPEG 实例
    jpeg = JPEG()

    # 调用 compress_and_store 方法处理图片并保存
    output_image_path = r"E:\AnimeSR\compare_5\banding\1111.png"
    jpeg.compress_and_store(input_image, output_image_path, 0)