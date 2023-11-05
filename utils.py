import random
import numpy as np
from PIL import Image
import torch
import os
import re
import SHA256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Miller-Rabin素数测试
def is_prime(n, k=5):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # 将n-1表示为2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # 进行Miller-Rabin测试
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


# 生成指定位数的大素数
def generate_large_prime(bits):
    while True:
        candidate = random.getrandbits(bits) | (1 << bits - 1) | 1
        if is_prime(candidate):
            return candidate


def gc(x, size):
    tensor_1 = torch.randint(low=-100, high=100, size=(size - 1,), device=device)
    tensor_2 = torch.tensor([x]).to(device)
    coefficient = torch.cat((tensor_2, tensor_1), dim=0)
    return coefficient


def generate_coefficients(x, size):
    coefficients = [gc(_, size) for _ in x]
    coefficients = torch.stack(coefficients)
    return coefficients


def generate_abscissa(s, num, shuffle=True):
    # 定义整数范围

    start_value = 1  # 指定起始值
    end_value = 56  # 指定结束值
    n = end_value - start_value + 1  # 计算整数范围
    sequence = None
    if shuffle:
        # 生成一个不重复的随机整数序列，范围是起始值到结束值
        sequence = start_value + torch.randperm(n, device=device)
    else:
        sequence = torch.arange(start_value, end_value, device=device)
    if sequence is None:
        print('error in generate_abscissa')
        return
    abscissas = [sequence[:num] for _ in s]
    abscissas = torch.stack(abscissas)
    return abscissas


def recovery(s, t, p):
    if len(s) < t:
        print('too less')
        return
    S = []
    # print(s.shape)
    for i in range(t):
        sigma = 1
        dem = 1
        for j in range(t):
            if s[j][0] != s[i][0]:
                sigma *= (s[j][0])
                dem *= (s[j][0] - s[i][0])
        # 保证sigma为0到255的正整数且是模值运算
        if s[i][1]*sigma % dem == 0:
            sigma = s[i][1] * sigma / dem
        else:
            x = dem
            dem = mod_inverse(dem % p, p)
            if dem is None:
                print(x, " ", p, " ", s[i][1]*sigma / x)
            sigma = s[i][1]*sigma*dem
        S.append(sigma)
    S = [_.view(1) for _ in S]
    S = torch.cat(S)
    # print(S.shape)
    result = torch.sum(S) % p
    # S.append(result)
    return result


def extended_gcd(a, m):
    if a == 0:
        return m, 0, 1
    gcd, x1, y1 = extended_gcd(m % a, a)
    x = y1 - (m // a) * x1
    y = x1
    return gcd, x, y


def mod_inverse(a, m):
    g, x, y = extended_gcd(a, m)
    # 如果最大公约数不等于1，表示模逆不存在
    if g != 1:
        return None
    else:
        # 计算模逆，确保它是正数
        x = (x % m + m) % m
        return x


# 分割图像成相应patches，返回PIL对象
def get_patch(image_path, patch_size):
    image = Image.open(image_path)

    # 获取图像的宽度和高度
    width, height = image.size
    # print(image.size)
    # 定义要分割的块的大小
    patch_width, patch_height = patch_size

    patches = []

    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            # 切割图像
            patch = image.crop((x, y, x + patch_width, y + patch_height))
            patches.append(patch)

    return patches


# 获取图像的路径
def get_images_from_dir(image_dir):
    if not (os.path.exists(image_dir) and os.path.isdir(image_dir)):
        print(image_dir)
        print("illegal patch dir in utils.get_images_from_dir")
        return
    all_contents = os.listdir(image_dir)
    image_list = [item for item in all_contents if os.path.isfile(os.path.join(image_dir, item).replace('\\', '/'))]
    # print(image_list)
    images = []
    for path in image_list:
        image_path = os.path.join(image_dir, path).replace('\\', '/')
        images.append(image_path)
    return images


# 生成patches图像并保存
def generate_patches(image_path, patch_dir=None, patch_size=None):
    if patch_dir is None:
        patch_dir = './patches'
    os.makedirs(patch_dir, exist_ok=True)
    if patch_size is None:
        print('please input the size of patch')
        return
    patch_list = get_patch(image_path, patch_size)
    index = 0
    for patch in patch_list:
        patch.save(patch_dir + '/' + str(index) + '.png')
        index += 1
    print(f'patches is save in {patch_dir}')
    return patch_dir


# 从patches重构原始图像
def reconstruct_image_from_dir(patch_dir, save_path):
    patches = get_images_from_dir(patch_dir)
    reconstruct_image_from_path_list(patches, save_path)


def reconstruct_image_from_path_list(patches, save_path):
    patch_num = len(patches)
    img = Image.open(patches[0])
    patch_width, patch_height = img.size
    scale = patch_width/patch_height
    image_range = patch_width*patch_height*patch_num
    original_height = (image_range/scale)**0.5
    original_width = image_range/original_height
    C = img.mode
    image = Image.new(C, (int(original_width), int(original_height)))
    for index, patch in enumerate(patches):
        patch_image = Image.open(patch)
        x = (index % (original_width // patch_width)) * patch_width
        y = (index // (original_width // patch_width)) * patch_height
        image.paste(patch_image, (int(x), int(y)))
    image.save(save_path)
    print(f'the reconstruct image is saved in {save_path}')
    return image


def find_common_factor(w, h):
    index = 0
    for i in range(2, 12):
        if w % i == 0 and h % i == 0:
            index = i
    return index


def extract_images_from_patches_out(patch_dir):
    if not (os.path.exists(patch_dir) and os.path.isdir(patch_dir)):
        print("illegal patch dir in utils.extract_images_from_patches_out")
        return
    patch_dir_content = os.listdir(patch_dir)
    patches_list = []
    for item in patch_dir_content:
        patches = get_images_from_dir(os.path.join(patch_dir, item, 'signature').replace('\\', '/'))
        if patches is None:
            break
        patches_list.append(patches)

    # 使用列表解析来转置嵌套列表
    reconstruct_patches_path = [[row[i] for row in patches_list] for i in range(len(patches_list[0]))]
    print(reconstruct_patches_path)
    return reconstruct_patches_path


def extract_images_from_patches_buffer(patch_dir):
    if not (os.path.exists(patch_dir) and os.path.isdir(patch_dir)):
        print("illegal patch dir in utils.extract_images_from_patches_out")
        return
    patch_dir_content = os.listdir(patch_dir)
    patches_list = []
    for item in patch_dir_content:
        patches = get_images_from_dir(os.path.join(patch_dir, item).replace('\\', '/'))
        if patches is None:
            return
        patches_list.append(patches)

    # 使用列表解析来转置嵌套列表
    reconstruct_patches_path = [[row[i] for row in patches_list] for i in range(len(patches_list[0]))]
    print(reconstruct_patches_path)
    return reconstruct_patches_path


if __name__ == '__main__':
    # just to test something
    src = './resources/... .png'
    out_src = './back/... .png'
    image_src = Image.open(src)
    image_out_src = Image.open(out_src)
    image_array_1 = np.array(image_src)
    image_array_2 = np.array(image_out_src)
    diff_indices = np.where(image_array_1 != image_array_2)
    print(diff_indices)
