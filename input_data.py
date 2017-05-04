
import tensorflow as tf
import numpy as np
import os


def read_cifar10(data_dir, is_train, batch_size, shuffle):
    """
    Read cifar10 data
    :param data_dir: data directory
    :param is_train: input train data or test data
    :param batch_size: batch size
    :param shuffle: whether shuffle the data
    :return: label: 1D tensor, [batch_size, n_classes], one-hot coding, tf.int32
             images: 4D tensor, [batch_size, width, height, 3], tf.float32
    """

    img_width = 32
    img_height = 32
    img_channel = 3
    label_bytes = 1
    image_bytes = img_width * img_height * img_channel

    with tf.name_scope('input'):

        data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

        if is_train:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % ii) for ii in np.arange(1, 6)]
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]

        filename_queue = tf.train.input_producer(filenames)
        reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)
        key, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.slice(record_bytes, [0], [label_bytes])
        label = tf.cast(label, tf.int32)

        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
        image_raw = tf.reshape(image_raw, [img_channel, img_height, img_width])
        image = tf.transpose(image_raw, (1, 2, 0))  # convert D/H/W -> H/W/D
        image = tf.cast(image, tf.float32)

        # normalization: (x - mean) / var
        image = tf.image.per_image_standardization(image)

        # tf.train.shuffle_batch() Args:
        #
        # tensors: The list or dictionary of tensors to enqueue.
        # batch_size: The new batch size pulled from the queue.
        # capacity: An integer. The maximum number of elements in the queue.
        # min_after_dequeue: Minimum number elements in the queue after a dequeue,
        #                    used to ensure a level of mixing of elements.
        # num_threads: The number of threads enqueuing tensor_list.
        if shuffle:
            images, label_batch = tf.train.shuffle_batch([image, label],
                                                         batch_size=batch_size,
                                                         capacity=20000,
                                                         min_after_dequeue=3000,
                                                         num_threads=64)
        else:
            images, label_batch = tf.train.batch([image, label],
                                                 batch_size=batch_size,
                                                 capacity=2000,
                                                 num_threads=64)
        # one-hot coding
        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth=n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])

        return images, label_batch



