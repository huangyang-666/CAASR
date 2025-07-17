import sys, os, random
import cv2, torch
from multiprocessing import Process, Queue

root_path = os.path.abspath('.')
sys.path.append(root_path)
# Import files from the local folder
from opt import opt
from degradation.ESR.utils import tensor2np, np2tensor

import numpy as np
import cv2


class JPEG:
    def __init__(self) -> None:
        # Initialize with optional properties
        pass

    # 通过转换图像到HSV颜色空间（色调H，饱和度S，亮度V）来调整饱和度和对比度。通过修改S和V通道的值来实现这一点。saturation_scale
    # 和contrast_scale参数控制饱和度和对比度的调整幅度。这里饱和度降低了20%，对比度提高了20%
    # todo 多少合适？
    def adjust_saturation_and_contrast(self, image, saturation_scale=1.0, contrast_scale=2.0):  # [0.5,2.0]  [0.5,2.0]
        """ Adjust the saturation and contrast of the image """
        # Convert image from BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split into channels
        h, s, v = cv2.split(hsv_image)

        # Adjust saturation, ensuring no overflow
        s = cv2.multiply(s, saturation_scale)
        s = np.clip(s, 0, 255)

        # Adjust value channel for contrast
        v = cv2.multiply(v, contrast_scale)
        v = np.clip(v, 0, 255)

        # Merge channels back to an HSV image
        hsv_image = cv2.merge([h, s, v])

        # Convert image back to BGR
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return adjusted_image

    def compress_and_store(self, np_frames, store_path, idx):
        ''' Compress and Store the whole batch as JPEG
        Args:
            np_frames (numpy):      The numpy format of the data
            store_path (str):       The store path
        Return:
            None
        '''

        # Preparation
        single_frame = np_frames

        # Simulate saturation and contrast changes
        single_frame = self.adjust_saturation_and_contrast(single_frame, saturation_scale=0.8, contrast_scale=1.0)

        # Compress as JPEG
        jpeg_quality = random.randint(*opt['jpeg_quality_range2'])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode('.jpg', single_frame, encode_param)
        decimg = cv2.imdecode(encimg, 1)

        # Store the image with quality
        cv2.imwrite(store_path, decimg)


if __name__ == '__main__':
    # 读取输入图片
    input_image_path = "E:\AnimeSR\compare_5\input\haizeiwang_3_12.png"
    input_image = cv2.imread(input_image_path)

    # 创建 JPEG 实例
    jpeg = JPEG()

    # 调用 compress_and_store 方法处理图片并保存
    output_image_path = "E:\AnimeSR\compare_5\input\output_image.jpg"
    jpeg.compress_and_store(input_image, output_image_path, 0)
