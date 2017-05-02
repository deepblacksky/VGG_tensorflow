"""
This is a tool for tensorflow neural network.
It include conv function, pool function, etc.
"""
import tensorflow as tf
import numpy as np


def conv(layer_name, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=True):
    """
    Convolution op wrapper, the Activation id ReLU
    :param layer_name: layer name, eg: conv1, conv2, ...
    :param x: input tensor, size = [batch_size, height, weight, channels]
    :param out_channels: number of output channel (convolution kernel)
    :param kernel_size: convolution kernel size, VGG use [3,3]
    :param stride:
    :param is_pretrain:
    :return:
    """


