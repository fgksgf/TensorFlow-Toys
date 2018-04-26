# retrain.py

import os
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
import cifar10
from cache import transfer_values_cache
from cifar10 import num_classes
from inception.inception import Inception

# 载入CIFAR-10数据集类别名称
# ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
class_names = cifar10.load_class_names()

# 载入训练集，返回图像、整形分类号码、以及用One-Hot编码的分类号数组
images_train, cls_train, labels_train = cifar10.load_training_data()
# 载入测试集，返回图像、整形分类号码、以及用One-Hot编码的分类号数组
images_test, cls_test, labels_test = cifar10.load_test_data()

# 设置训练集和测试集缓存文件的目录
file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

# 加载Inception V3模型
model = Inception()

print("Processing transfer-learning transfer-values for training-images ...")

# 如果训练数据的transfer-values已经计算过，则从文件中加载出来；否则计算它们并保存为缓存文件
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_train,
                                              model=model)

# 检查transfer-values的数组大小: (50000, 2048)
# 在训练集中有50,000张图像，每张图像有2048个transfer-values
print('The shape of transfer-values for training-images: ', transfer_values_train.shape)

print("Processing transfer-learning transfer-values for test-images ...")

# 如果测试数据的transfer-values已经计算过，则从文件中加载出来；否则计算它们并保存为缓存文件
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             images=images_test,
                                             model=model)

# 检查transfer-values的数组大小: (10000, 2048)
# 在测试集中有10,000张图像，每张图像有2048个transfer-values
print('The shape of transfer-values for test-images: ', transfer_values_test.shape)

###################################################################################
# 在TensorFlow中创建一个新的神经网络
# 这个网络会把Inception模型中的transfer-values作为输入
# 然后输出CIFAR-10图像的预测类别

# transfer-values的数组长度
transfer_len = model.transfer_len

# 为输入的transfer-values创建一个placeholder变量
# 变量的形状是[None, transfer_len]
# None表示它的输入数组包含任意数量的样本，每个样本元素个数为2048，即transfer_len
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')

# 为输入图像的真实类型标签定义另外一个placeholder变量
# 这是One-Hot编码的数组，包含10个元素，每个元素代表了数据集中的一种可能类别
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

# 计算代表真实类别的整形数字,即取y_true这个One-Hot编码的数组的最大值的位置索引
y_true_cls = tf.argmax(y_true, dimension=1)

# 全连接层
layer_fc = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)

# softmax层，包含10个神经元
logits = tf.layers.dense(inputs=layer_fc, units=10)
y_pred = tf.nn.softmax(logits, name="softmax_tensor")

# 计算交叉熵
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
# 以交叉熵作为损失函数
loss = tf.reduce_mean(cross_entropy)

# 创建一个变量来记录当前优化迭代的次数。
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

# 优化神经网络的方法：adam方法，学习率为0.0001，目标为最小化损失函数 90.3
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

# 输出层y_pred所表示的类编号，取y_pred中概率最大值的索引
y_pred_cls = tf.argmax(y_pred, dimension=1)

# 创建一个布尔向量，表示每张图像的真实类别是否与预测类别相同。
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# 将布尔值向量类型转换成浮点型向量，False就变成0，True变成1
# 然后计算这些值的平均数，以此来计算分类的准确度。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 每次同时输入多少张图的transfer-values
train_batch_size = 64

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256


# 用来从训练集中选择随机batch的transfer-values
def random_batch():
    # 训练集中的图片个数
    num_images = len(transfer_values_train)

    # 创建随机的索引
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # 使用该索引来选择随机的x和y值
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch


# 用来执行一定数量的优化迭代，以此来逐渐改善网络层的变量
# 在每次迭代中，会从训练集中选择新的一批数据，然后TensorFlow在这些训练样本上执行优化
# 每500次迭代会打印出进度
def optimize(num_iterations):
    # 执行优化开始时间
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()

        # 将数据传入TensorFlow对应的占位符中
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # 运行optimizer并获取全局步数
        i_global, _ = session.run([global_step, optimizer], feed_dict=feed_dict_train)

        # 每迭代500次输出一次状态信息
        if (i_global % 500 == 0) or (i == num_iterations - 1):
            # 计算损失函数
            batch_loss = session.run(loss, feed_dict=feed_dict_train)

            msg = "Global Step: {0:>6}, Training Batch Loss: {1:>6.4}"
            print(msg.format(i_global, batch_loss))

    # 优化结束时间
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# 用来计算图像的预测类别，同时返回一个代表每张图像分类是否正确的布尔数组
def predict_cls(transfer_values, labels, cls_true):
    # Number of images.
    num_images = len(transfer_values)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


# 计算测试集上的预测类别。
def predict_cls_test():
    return predict_cls(transfer_values=transfer_values_test,
                       labels=labels_test,
                       cls_true=cls_test)


# 计算给定布尔数组的分类准确率，布尔数组表示每张图像是否被正确分类
# 比如， cls_accuracy([True, True, False, False, False]) = 2/5 = 0.4。
def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()


# 用来打印测试集上的分类准确率。
def print_test_accuracy():
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))


if __name__ == '__main__':
    # 创建TensorFlow会话,用来运行图
    session = tf.Session()

    # 需要在开始优化weights和biases变量之前对它们进行初始化。
    session.run(tf.global_variables_initializer())

    optimize(num_iterations=20000)
    print_test_accuracy()
    model.close()
    session.close()
