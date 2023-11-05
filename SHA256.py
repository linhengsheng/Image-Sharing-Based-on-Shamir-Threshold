import hashlib
from PIL import Image
import rsa  # 使用RSA算法进行数字签名


# 生成图像哈希值，只对像素信息进行签名
def calculate_hash(image_path):
    # with open(image_path, 'rb') as f:
    #     image_data = f.read()

    image = Image.open(image_path)
    image_data = image.tobytes()
    return hashlib.sha256(image_data).hexdigest()


# 生成数字签名
def sign_image(image_path, private_key_path):
    image_hash = calculate_hash(image_path)
    with open(private_key_path, 'rb') as f:
        private_key = rsa.PrivateKey.load_pkcs1(f.read())
        signature = rsa.sign(image_hash.encode('utf-8'), private_key, 'SHA-256')
    return signature


# 将数字签名嵌入到图像中
def embed_signature_in_image(image_path, signature, output_path):
    image_hash = calculate_hash(image_path).encode('utf-8')
    # combined_data = f"\r\n{image_hash}\r\n{signature}"  # 将哈希值和签名以换行符分隔的方式组合在一起
    combined_data = b''.join([b'\r\n', image_hash, b'\r\n', signature])

    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    with open(output_path, 'wb') as output_file:
        output_file.write(image_data)
    with open(output_path, 'ab') as output_file:
        output_file.write(combined_data)
    print(f'signature is embeded in image, the image_out is save in {output_path}')


# 从图像中提取签名
def extract_signature_from_image(image_path):
    print(image_path)
    with open(image_path, 'rb') as f:
        image_data = f.read()
        index = 0
        for i in range(len(image_data)):
            if image_data[i] == 0x0D and image_data[i+1] == 0x0A:
                index = i+2
        signature = image_data[index:]
        return signature


# 存储数字签名
def save_signature(signature, signature_path):
    with open(signature_path, 'wb') as f:
        f.write(signature)


# 验证数字签名
def verify_signature(image_path, public_key_path, signature_path):
    image_hash = calculate_hash(image_path)
    with open(public_key_path, 'rb') as f:
        public_key = rsa.PublicKey.load_pkcs1(f.read())
        with open(signature_path, 'rb') as f2:
            signature = f2.read()
            try:
                rsa.verify(image_hash.encode('utf-8'), signature, public_key)
                print("数字签名验证通过")
            except rsa.VerificationError:
                print("数字签名验证失败")


if __name__ == '__main__':
    image_path = './resources/logo.png'
    private_key_path = 'private_key.pem'
    public_key_path = 'public_key.pem'
    signature_path = 'image_signature'

    # 生成数字签名并保存
    signature = sign_image(image_path, private_key_path)
    save_signature(signature, signature_path)

    # 验证数字签名
    verify_signature(image_path, public_key_path, signature_path)
