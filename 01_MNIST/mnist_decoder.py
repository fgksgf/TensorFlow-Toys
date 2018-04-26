# 将MNIST手写数字数据文件转换为numpy数组。

import numpy as np
import struct
import matplotlib.pyplot as plt

# 训练集文件存放位置
train_images_idx3_ubyte_file = './data/train-images.idx3-ubyte'
# 训练集标签文件存放位置
train_labels_idx1_ubyte_file = './data/train-labels.idx1-ubyte'

# 测试集文件存放位置
test_images_idx3_ubyte_file = './data/t10k-images.idx3-ubyte'
# 测试集标签文件存放位置
test_labels_idx1_ubyte_file = './data/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3格式文件
    :param idx3_ubyte_file: idx3文件路径
    :return: numpy数组，数据类型为float32
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols), dtype=np.float32)
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1格式文件
    :param idx1_ubyte_file: idx1文件路径
    :return: numpy数组，数据类型为int32
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images, dtype=np.int32)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    加载训练集数据
    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    加载训练集标签
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    加载测试集数据
    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    加载测试集标签
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def test():
    """
    检验MNIST数据集是否正确解析
    """
    test_images = load_test_images()
    test_labels = load_test_labels()

    # 查看测试集中前十个数据及其标签是否读取正确
    for i in range(10):
        print(test_labels[i])
        plt.imshow(test_images[i], cmap='gray')
        plt.show()


if __name__ == '__main__':
    test()
