import sys, os, random
import cv2, torch
import numpy as np

root_path = os.path.abspath('.')
sys.path.append(root_path)
from degradation.ESR.utils import tensor2np, np2tensor
from opt import opt


# 为每个颜色通道（蓝色B、绿色G、红色R）生成一个随机的色彩偏移量,并将这些偏移量应用到各自的通道上，偏移的强度通过shift——intensity控制
# todo 需要测试强度值究竟该多少好呢
def simulate_color_shift(tensor_frames, shift_intensity):
    """ Simulate color shift by randomly altering the color balance """
    # Define shift intensity
    # shift_intensity = 25  # You can adjust this value for more or less intensity   10肉眼能看出来，25非常离谱

    single_frame = tensor2np(tensor_frames)

    # Randomly generate color shifts for each channel
    b_shift = random.randint(-shift_intensity, shift_intensity)
    g_shift = random.randint(-shift_intensity, shift_intensity)
    r_shift = random.randint(-shift_intensity, shift_intensity)

    # Split image into color channels
    b, g, r = cv2.split(single_frame)

    # Apply color shift
    b = np.clip(b + b_shift, 0, 255).astype(np.uint8)
    g = np.clip(g + g_shift, 0, 255).astype(np.uint8)
    r = np.clip(r + r_shift, 0, 255).astype(np.uint8)

    # Merge channels back to an image
    shifted_image = cv2.merge([b, g, r])
    # decimg = np.asarray(shifted_image)
    #
    # result = np2tensor(decimg)

    return shifted_image


def simulate_banding(tensor_frames, levels=2):
    """ Simulate color banding effect by reducing color levels and then restoring """
    single_frame = tensor2np(tensor_frames)

    banding_image = np.floor(single_frame / levels) * (levels)

    decimg = np.clip(banding_image, 0, 255).astype(np.uint8)

    # result = np2tensor(decimg)

    return decimg


def simulate_blocking(tensor_frames, frequency_cutoff_ratio=0.9):
    """ Simulate blocking effect by artificially enhancing block edges in 8x8 grid for YCbCr color channels
        frequency_cutoff_ratio controls the intensity of the blocking effect by defining how much
        of the high-frequency components are set to zero in the Cb and Cr channels.
    """
    image = tensor2np(tensor_frames)
    # print(image.shape)
    block_size = 8
    # Convert RGB to YCbCr
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    # Initialize output image
    output_image = ycbcr_image.copy()

    # Process Cb and Cr channels
    for channel in range(1, 3):  # Cb and Cr channels
        for i in range(0, ycbcr_image.shape[0], block_size):
            for j in range(0, ycbcr_image.shape[1], block_size):
                block = ycbcr_image[i:i + block_size, j:j + block_size, channel]
                # print(f"in===>{block}")
                dct_block = cv2.dct(block.astype(np.float32))  # Apply DCT
                mask = np.ones_like(block)  # 创建与block相同形状的mask
                mask[int(block_size * frequency_cutoff_ratio):, int(block_size * frequency_cutoff_ratio):] = 0
                dct_block *= mask
                idct_block = cv2.idct(dct_block)
                output_image[i:i + block_size, j:j + block_size, channel] = idct_block

    # Convert back to RGB
    result_image = cv2.cvtColor(output_image, cv2.COLOR_YCrCb2RGB)
    decimg = np.clip(result_image, 0, 255).astype(np.uint8)

    return decimg


if __name__ == '__main__':
    # 加载图像
    img_path = r"/home/zyj/AnimeBetter_hy/project2_AnimeSR/APISR-main-ablations-2d-allin/0_rebuttle/input/00000003.png"
    img = cv2.imread(img_path)

    # 将图像转换为 PyTorch 张量
    tensor_img = torch.tensor(img.transpose(2, 0, 1)).float() / 255.0  # 将通道维度调整为第一个维度，并进行归一化

    # 设置色彩偏移强度
    # shift_intensity = 10

    # 应用 simulate_color_shift 方法
    # shifted_img = simulate_color_shift(tensor_img, 20)
    shifted_img = simulate_banding(tensor_img, 15)
    # shifted_img = simulate_blocking(tensor_img, 0.2)

    # 保存生成的效果图
    # output_path = r"/home/zyj/AnimeBetter_hy/project2_AnimeSR/APISR-main-ablations-2d-allin/0_rebuttle/output/color_shift.png"
    output_path = r"/home/zyj/AnimeBetter_hy/project2_AnimeSR/APISR-main-ablations-2d-allin/0_rebuttle/output/color_banding.png"
    # output_path = r"/home/zyj/AnimeBetter_hy/project2_AnimeSR/APISR-main-ablations-2d-allin/0_rebuttle/output/color_blocking.png"
    cv2.imwrite(output_path, shifted_img)

    print("Generated image saved at:", output_path)
