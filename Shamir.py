import numpy as np
import torch
from utils import device
import utils
from PIL import Image
import os
import re
import SHA256
import rsa


private_key_path = 'private_key.pem'
public_key_path = 'public_key.pem'


class Shamir:
    def __init__(self, n: int, t: int):
        if t <= 0:
            print('wrong t')
            return
        self.n = n
        self.t = t
        self.shape = None
        self.flatten = torch.nn.Flatten()
        self.p = 256    # p为257 可能导致 max:256； p为256可能导致模逆不可求

    # forward 和 recover是[secret_num, share_num|threshold_num, 2]tensor的科学计算，分解与恢复子秘密
    def forward(self, s, shuffle=True):
        s = s.to(device)
        # 系数生成
        s = s.view(-1)
        coefficients = utils.generate_coefficients(s, self.t)
        abscissas = utils.generate_abscissa(s, self.n, shuffle)
        print(coefficients)
        print(abscissas)
        # 秘密数生成
        secrets_tensors = []
        for coefficient, abscissa in zip(coefficients, abscissas):
            secrets_tensor = []
            for i in range(self.n):
                x = abscissa[i]
                y = (torch.sum(coefficient * (x ** torch.arange(len(coefficient), device=device)))) % self.p
                result = torch.stack((x, y))
                secrets_tensor.append(result)
            secrets_tensor = torch.stack(secrets_tensor)
            secrets_tensors.append(secrets_tensor)
        secrets_tensors = torch.stack(secrets_tensors)
        print(f'secrets_tensors: dtype={secrets_tensors.dtype}  shape{secrets_tensors.shape}')
        return secrets_tensors

    def recover(self, s):
        s = s.to(device)
        secrets = []
        for _ in s:
            secret = utils.recovery(_, self.t, self.p)
            if secret is None:
                return
            secrets.append(secret)
        secrets = [_.view(1) for _ in secrets]
        secrets = torch.cat(secrets)
        return secrets

    # 生成子秘密图像，index保存在图片的名称中
    def image_in(self, image_path, save_dir, shuffle=False):
        signature = SHA256.sign_image(image_path, private_key_path)
        save_name = image_path.split('/')[-1].split('.')[0]
        save_dir = save_dir + '/' + save_name
        os.makedirs(save_dir, exist_ok=True)
        print('the pictures will be saved in: ', save_dir)
        image = Image.open(image_path)
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array)
        self.shape = image_tensor.shape
        # 图片形状
        print(self.shape)
        # 子秘密数生成
        secrets_tensor = self.forward(image_tensor, shuffle)
        max_value = torch.max(secrets_tensor)
        print(f"max:{max_value.item()}")
        self.secrets_tensor_to_images(secrets_tensor, save_dir, signature)

    # 输出图像要使用png无损压缩存储，否则会出现原图不匹配的错误
    def image_out(self, save_dir, save_path):
        secrets_tensor, signature_path = self.images_to_secrets_tensor(save_dir)
        image_tensor = self.recover(secrets_tensor)
        # print(image_tensor.shape)
        # print(image_tensor.dtype)
        image_array = image_tensor.cpu().numpy()
        image_array = image_array.reshape(self.shape)
        image_array = image_array.astype(np.uint8)
        image = Image.fromarray(image_array)
        image.save(save_path)
        print(f'image is save in {save_path}')
        self.image_verify(signature_path, save_path)

    # image_in; 把[secret_num, share_num|threshold_num, 2]的可执行tensor类型分解成
    # [share_num|threshold_num, H, W, C]numpy数组，最低维度是secret, index作为函数名称存放在路径中
    def secrets_tensor_to_images(self, secrets_tensors, save_dir, signature):
        out_common = save_dir + '/common'
        out_signature = save_dir + '/signature'
        os.makedirs(out_common, exist_ok=True)
        os.makedirs(out_signature, exist_ok=True)
        secret_arrays = secrets_tensors.cpu().numpy()
        secret_arrays = np.transpose(secret_arrays, axes=[1, 0, 2])
        image_arrays = []
        for secret_array in secret_arrays:
            secret_array = np.transpose(secret_array, axes=[1, 0])
            index = secret_array[0][0]
            image = secret_array[1].reshape(self.shape).astype(np.uint8)
            image_arrays.append(image)
            image = Image.fromarray(image)
            image.save(f'{out_common}/common_{index}.png')
            SHA256.embed_signature_in_image(f'{out_common}/common_{index}.png',
                                            signature, f'{out_signature}/signature_{index}.png')
        print(f'images is save in {save_dir}')

    # image_out使用;把子秘密图像转化成[secret_num, share_num|threshold_num, 2]的可恢复tensor类型, 最低度维数据代表[x, s]
    def images_to_secrets_tensor(self, save_dir):
        all_contents = os.listdir(save_dir)
        image_list = [os.path.join(save_dir, item).replace('\\', '/') for item in all_contents if os.path.isfile(os.path.join(save_dir, item).replace('\\', '/'))]
        # print(image_list)
        # secrets_arrays = []
        # ret_path = None
        # for path in image_list:
        #     ret_path = os.path.join(save_dir, path).replace('\\', '/')
        #     image = Image.open(os.path.join(save_dir, path).replace('\\', '/'))
        #     image_array = np.array(image)
        #     self.shape = image_array.shape
        #     image_array = image_array.flatten()
        #     match = re.search(r'\d+', path)
        #     if match:
        #         index = match.group()
        #         index_array = np.full(self.shape, int(index))
        #         index_array = index_array.flatten()
        #     else:
        #         return
        #     secrets_array = np.stack((index_array, image_array))
        #     secrets_array = np.transpose(secrets_array, axes=[1, 0])
        #     secrets_arrays.append(secrets_array)
        # secrets_arrays = np.stack(secrets_arrays)
        secrets_arrays, ret_path = self.get_secrets_array_from_image_list(image_list)
        secrets_arrays = np.transpose(secrets_arrays, axes=[1, 0, 2])
        # print(secrets_arrays)
        # print(secrets_arrays.shape)
        secrets_tensor = torch.from_numpy(secrets_arrays)
        return secrets_tensor, ret_path

    # 将指定路由列表内的images转化成numpy数组（子秘密图像的重构过程）
    # 重构的index是子秘密图像的名称（image_out）或者根目录index(large_image_out）
    def get_secrets_array_from_image_list(self, image_list):
        secrets_arrays = []
        ret_path = None
        for path in image_list:
            ret_path = path
            image = Image.open(path)
            image_array = np.array(image)
            self.shape = image_array.shape
            image_array = image_array.flatten()
            match = re.search(r'\d+', path)
            if match:
                index = match.group(0)
                index_array = np.full(self.shape, int(index))
                index_array = index_array.flatten()
            else:
                print('the index is in error')
                return
            secrets_array = np.stack((index_array, image_array))
            secrets_array = np.transpose(secrets_array, axes=[1, 0])
            # print(secrets_array)
            secrets_arrays.append(secrets_array)
        secrets_arrays = np.stack(secrets_arrays)
        return secrets_arrays, ret_path

    def large_image_in(self, image_path, save_dir, patch_save_dir):
        src_image = Image.open(image_path)
        w, h = src_image.size
        gcd, x, y = utils.extended_gcd(w, h)
        if gcd < 2:
            print('cannot cut the image')
        else:
            s = utils.find_common_factor(w, h)
            patch_size = (w//s, h//s)
            utils.generate_patches(image_path, patch_save_dir, patch_size)   # 分割图像，保存在patch_dir
            paths = utils.get_images_from_dir(patch_save_dir)   # 获取patch图像源路径
            for index, path in enumerate(paths):
                print(f'creating secrets of {path}')
                self.image_in(path, save_dir, False)   # 生成子秘密图像patches
            self.large_image_reconstruct(image_path, save_dir, True)

    def large_image_out(self, save_dir, save_path, patches_buffer=None):
        secrets = utils.get_images_from_dir(save_dir)
        image_name, image_type = save_path.split("/")[-1].split(".")
        if image_type != "png":
            print('save image must be type "png"')
            save_path = save_path.replace(image_type, "png")
        if patches_buffer is None:
            patches_buffer = './patches_out/' + image_name
        # print(save_path)  ./back/a.png
        # print(patches_buffer) ./patches_out/a
        signature_path = None
        for index, secret_image in enumerate(secrets):
            if index == 0:
                signature_path = secret_image
            if index >= self.t:
                break
            patches_out_dir = patches_buffer + f'/{index + 1}'
            os.makedirs(patches_out_dir, exist_ok=True)
            src_image = Image.open(secret_image)
            w, h = src_image.size
            gcd, x, y = utils.extended_gcd(w, h)
            if gcd < 2:
                print('cannot cut the image')
            else:
                s = utils.find_common_factor(w, h)
                patch_size = (w // s, h // s)
                utils.generate_patches(secret_image, patches_out_dir, patch_size)  # 分割图像，保存在patch_dir
        image_list = utils.extract_images_from_patches_buffer(patches_buffer)
        for index, images in enumerate(image_list):
            secrets_arrays, _ = self.get_secrets_array_from_image_list(images)
            secrets_arrays = np.transpose(secrets_arrays, axes=[1, 0, 2])
            secrets_tensor = torch.from_numpy(secrets_arrays)
            image_tensor = self.recover(secrets_tensor)
            # print(image_tensor.dtype)
            image_array = image_tensor.cpu().numpy()
            image_array = image_array.reshape(self.shape)
            image_array = image_array.astype(np.uint8)
            image = Image.fromarray(image_array)
            patch_save_path = patch_dir + "/" + str(index) + ".png"
            image.save(patch_save_path)
        img_back = utils.reconstruct_image_from_dir(patch_dir, save_path)
        self.image_verify(signature_path, save_path)

    @staticmethod
    # 将数字签名嵌入图像中并保存
    def signature_in_image(image_path):
        signature = SHA256.sign_image(image_path, private_key_path)
        os.makedirs(signature_out_dir, exist_ok=True)
        output_path = f'{signature_out_dir}/{(image_path.split("/")[-1]).split(".")[0]}.png'
        SHA256.embed_signature_in_image(image_path, signature, output_path)
        return output_path

    @staticmethod
    # 验证图像的完整性与真实性
    def image_verify(output_path, image_path):
        print(f'signature_path: {output_path}\nimage_to_verify: {image_path}')
        extracted_signature = SHA256.extract_signature_from_image(output_path)
        image_hash = SHA256.calculate_hash(image_path)
        # print(type(extracted_signature))
        # print(extracted_signature)
        with open(public_key_path, 'rb') as f:
            public_key = rsa.PublicKey.load_pkcs1(f.read())
        try:
            rsa.verify(image_hash.encode('utf-8'), extracted_signature, public_key)
            print("数字签名验证通过")
        except rsa.VerificationError:
            print("数字签名验证失败")

    # 大图像重构，输入源路径(图像名称)，重构image_dir(子秘密图像patches)中的图像， 输出到out_images中
    @staticmethod
    def large_image_reconstruct(image_path, image_dir, is_signature=True):
        print('large_image_reconstruct: ')
        print(image_path)
        print(image_dir)
        picture_name = image_path.split('/')[-1].split('.')[0]
        reconstruct_patches = utils.extract_images_from_patches_out(image_dir)
        common_save_dir = out_dir + '/' + picture_name + '/common'
        signature_save_dir = out_dir + '/' + picture_name + '/signature'
        os.makedirs(signature_save_dir, exist_ok=True)
        os.makedirs(common_save_dir, exist_ok=True)
        for i in range(share_num):
            if is_signature:
                save_path = signature_save_dir + f'/signature_{i + 1}.png'
                utils.reconstruct_image_from_path_list(reconstruct_patches[i], save_path)
                print('signature')
                signature = SHA256.sign_image(image_path, private_key_path)
                SHA256.embed_signature_in_image(save_path, signature, save_path)
                # shamir.image_verify(save_path, image_path)
            else:
                save_path = common_save_dir + f'/common_{i + 1}.png'
                utils.reconstruct_image_from_path_list(reconstruct_patches[i], save_path)


out_dir = './out_images'
patch_dir = './patches'
patch_out_dir = './patches_out'
signature_out_dir = './signature_images'

# 门限与子秘密配额
share_num = 6
threshold_num = 2
shamir = Shamir(share_num, threshold_num)
# 源路径
src = './resources/miku.png'
signature_dir = out_dir + '/' + src.split("/")[-1].split(".")[0] + '/signature'
out_src = f'./back/{src.split("/")[-1].split(".")[0]}.png'

# 生成图像的数字签名并验证
# out_path = shamir.signature_in_image(src)
# shamir.image_verify(out_path, out_src)

# 生成共享图像的子秘密图像并将原始图像的签名嵌入其中，恢复时验证
# shamir.image_in(src, out_dir)
# shamir.image_out(signature_dir, out_src)

# 如果图片太大，则分割成不同的patch再分解成子秘密图像
shamir.large_image_in(src, signature_dir, patch_dir)
shamir.large_image_out(signature_dir, out_src)
