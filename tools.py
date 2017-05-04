"""
This is a tool for tensorflow neural network.
It include conv function, pool function, etc.
"""
import tensorflow as tf
import numpy as np


def conv(layer_name, x, out_channels, kernel_size=None, stride=None, is_pretrain=True):
    """
    Convolution op wrapper, the Activation id ReLU
    :param layer_name: layer name, eg: conv1, conv2, ...
    :param x: input tensor, size = [batch_size, height, weight, channels]
    :param out_channels: number of output channel (convolution kernel)
    :param kernel_size: convolution kernel size, VGG use [3,3]
    :param stride: paper default = [1,1,1,1]
    :param is_pretrain: whether you need pre train, if you get parameter from other, you don not want to train again,
                        so trainable = false. if not trainable = true
    :return: 4D tensor
    """
    kernel_size = kernel_size if kernel_size else [3, 3]
    stride = stride if stride else [1, 1, 1, 1]

    in_channels = x.get_shape()[-1]

    with tf.variable_scope(layer_name):
        w = tf.get_variable(name="weights",
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=is_pretrain)
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0),
                            trainable=is_pretrain)
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')

        return x


def pool(layer_name, x, ksize=None, stride=None, is_max_pool=True):
    """
    Pooling op
    :param layer_name: layer name, eg:pool1, pool2,...
    :param x:input tensor
    :param ksize:pool kernel size, VGG paper use [1,2,2,1], the size of 2X2
    :param stride:stride size, VGG paper use [1,2,2,1]
    :param is_max_pool: default use max pool, if it is false, the we will use avg_pool
    :return: tensor
    """
    ksize = ksize if ksize else [1, 2, 2, 1]
    stride = stride if stride else [1, 2, 2, 1]

    if is_max_pool:
        x = tf.nn.max_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)

    return x


def batch_norm(x):
    """
    Batch Normalization (offset and scale is none). BN algorithm can improve train speed heavily.
    :param x: input tensor
    :return: norm tensor
    """
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)

    return x


def FC_layer(layer_name, x, out_nodes):
    """
    Wrapper for fully-connected layer with ReLU activation function
    :param layer_name: FC layer name, eg: 'FC1', 'FC2', ...
    :param x: input tensor
    :param out_nodes: number of neurons for FC layer
    :return: tensor
    """
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        # flatten into 1D
        flat_x = tf.reshape(x, [-1, size])

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)

        return x


def loss(logits, labels):
    """
    Compute loss
    :param logits: logits tensor, [batch_size, n_classes]
    :param labels: one_hot labels
    :return:
    """
    with tf.name_scope('loss') as scope:
        # use softmax_cross_entropy_with_logits(), so labels must be one-hot coding
        # if use sparse_softmax_cross_entropy_with_logits(), the labels not be one-hot
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss_temp = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope + '/loss', loss_temp)

        return loss_temp


def optimize(loss, learning_rate, global_step):
    """
    optimization, use Gradient Descent as default
    :param loss:
    :param learning_rate:
    :param global_step:
    :return:
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def accuracy(logits, labels):
    """
    Evaluate quality of the logits at predicting labels
    :param logits: logits tensor, [batch_size, n_class]
    :param labels: labels tensor
    :return:
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy_temp = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope+'/accuracy', accuracy_temp)

        return accuracy_temp


def num_correct_prediction(logits, labels):
    """
    Evaluate quality of the logits at predicting labels
    :param logits:
    :param labels:
    :return: number of correct prediction
    """
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)

    return n_correct


def load(data_path, session):
    """
    load the VGG16_pretrain parameters file
    :param data_path:
    :param session:
    :return:
    """
    data_dict = np.load(data_path, encoding='latin1').item()

    keys = sorted(data_dict.keys())
    for key in keys:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))


def load_with_skip(data_path, session, skip_layer):
    """
    Only load some layer parameters
    :param data_path:
    :param session:
    :param skip_layer:
    :return:
    """
    data_dict = np.load(data_path, encoding='latin1').item()

    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))


def test_load():
    """
    test load vgg16.npy, print the shape of data
    :return:
    """
    data_path = './/VGG16_pretrain//vgg16.npy'

    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)


def print_all_variables(train_only=True):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)
    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try:        # TF1.0
            t_vars = tf.global_variables()
        except:     # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))






