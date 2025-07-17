import sys, os, random
import cv2, torch
from multiprocessing import Process, Queue

root_path = os.path.abspath('.')
sys.path.append(root_path)
# Import files from the local folder
from opt import opt
from degradation.ESR.utils import tensor2np, np2tensor


class JPEG:
    def __init__(self) -> None:
        # Initialize with optional properties
        pass
    #为每个颜色通道（蓝色B、绿色G、红色R）生成一个随机的色彩偏移量,并将这些偏移量应用到各自的通道上，偏移的强度通过shift——intensity控制
    #todo 需要测试强度值究竟该多少好呢
    def simulate_color_shift(self, image):
        """ Simulate color shift by randomly altering the color balance """
        # Define shift intensity
        shift_intensity = 10  # You can adjust this value for more or less intensity   10肉眼能看出来，25非常离谱

        # Randomly generate color shifts for each channel
        b_shift = random.randint(-shift_intensity, shift_intensity)
        g_shift = random.randint(-shift_intensity, shift_intensity)
        r_shift = random.randint(-shift_intensity, shift_intensity)

        # Split image into color channels
        b, g, r = cv2.split(image)

        # Apply color shift
        b = cv2.add(b, b_shift)
        g = cv2.add(g, g_shift)
        r = cv2.add(r, r_shift)

        # Merge channels back to an image
        shifted_image = cv2.merge([b, g, r])
        return shifted_image


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

        # Simulate color shift  ---> 2
        single_frame = self.simulate_color_shift(single_frame)
        # single_frame = self.simulate_color_shift(single_frame)

        # Compress as JPEG
        jpeg_quality = random.randint(*opt['jpeg_quality_range2'])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode('.jpg', single_frame, encode_param)
        decimg = cv2.imdecode(encimg, 1)

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

