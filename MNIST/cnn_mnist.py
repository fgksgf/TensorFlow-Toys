import tensorflow as tf

from MNIST.mnist_decoder import load_train_images, load_train_labels, load_test_images, load_test_labels

# 设置记录日志内容的阙值
tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """CNN 模型函数"""
    # 输入层
    # 每一条训练数据表示为28 * 28 * 1的向量，因为图片只有一种颜色
    # -1表示使用全部的训练数据
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # 第一层卷积层, 输出为28 x 28 x 32的张量，因为有32个卷积核
    conv1 = tf.layers.conv2d(
        inputs=input_layer,  # 传入输入层
        filters=32,  # 卷积核个数为32
        kernel_size=[5, 5],  # 卷积核大小为5 x 5
        padding="same",  # same表示使用零填充来保持该层的输入和输出尺寸一致，即28 x 28
        activation=tf.nn.relu)  # 激活函数ReLu

    # 第一层池化层，最大池，大小为2 x 2， 步长为2
    # 输出为 14 x 14 x 32大小的张量
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二层卷积层和第二层池化层
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 将7 * 7 * 64的张量转置为3136 x 1的张量
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # 密集层，有1024个神经元和ReLU激活函数
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # 为防止过拟合，任何数据在训练期间有40%的概率被丢弃
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits层，它会返回我们预测的原始值
    # 包含10个神经元（每个目标类为0-9）的密集层，并使用线性激活（默认）
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # 0-9中取概率最大的类别
        "classes": tf.argmax(input=logits, axis=1),
        # 0-9每一类对应的概率
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 计算损失函数 (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # 使用0.001的学习率和随机梯度下降作为优化算法
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # 使用正确率来评估模型
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # 加载训练集和测试集
    train_data = load_train_images()
    train_labels = load_train_labels()
    eval_data = load_test_images()
    eval_labels = load_test_labels()

    # 创建一个Estimator类，用来训练和评估tensorflow的模型
    # 传入上面创建的CNN模型函数，以及模型要被保存的目录
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./tmp/mnist_convnet_model")

    # 因为训练需要一定的时间，为了更好地跟踪训练的进度，需要定时输出一些信息来帮助我们了解模型的状况
    # 设置一个字典张量来决定要需要记录的数据
    # 这里选择记录的是每一次预测时，预测结果为0-9各自的概率
    tensors_to_log = {"probabilities": "softmax_tensor"}

    # 将该张量传给日志钩子，使其每迭代50次记录一次概率
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # 训练模型
    # 首先，创建一个训练输入函数，传入训练数据和标签
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,  # 每一步使用100条数据进行训练
        num_epochs=None,  # 模型将一直训练直到指定的步数达到
        shuffle=True)  # 打乱训练数据的顺序

    # 调用模型的train方法
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=10000,  # 模型将总共训练20000步
        hooks=[logging_hook])  # 传入事件钩子

    # 评估模型并打印最终结果
    # 创建一个评估输入函数，传入评估数据集
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,  # 对模型评估一次
        shuffle=False)  # 不打乱测试集数据顺序

    # 调用模型的evaluate方法，传入评估输入函数
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    # 打印评估结果
    print(eval_results)


if __name__ == "__main__":
    # 执行main方法
    tf.app.run()
