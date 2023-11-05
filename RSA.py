import rsa


# 生成RSA密钥对
def generate_rsa_key_pair(bit_length=2048):
    (public_key, private_key) = rsa.newkeys(bit_length)
    return public_key, private_key


# 保存公钥和私钥到文件
def save_rsa_key_to_file(key, filename):
    with open(filename, 'wb') as f:
        f.write(key.save_pkcs1())


if __name__ == '__main__':
    # 生成RSA密钥对
    public_key, private_key = generate_rsa_key_pair()

    # 保存公钥和私钥到文件
    save_rsa_key_to_file(public_key, 'public_key.pem')
    save_rsa_key_to_file(private_key, 'private_key.pem')

    print("RSA密钥对已生成并保存到public_key.pem和private_key.pem文件中。")
