# DATA:
# 1. cifar10(binary version):https://www.cs.toronto.edu/~kriz/cifar.html
# 2. pratrained weights (vgg16.npy):https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM

# TO Train and test:
# 0. get data ready, get paths ready !!!
# 1. run training_and_val.py and call train() in the console
# 2. call evaluate() in the console to test

import os
import os.path

import math
import numpy as np
import tensorflow as tf

import input_data
import VGG
import tools


IMG_W = 32
IMG_H = 32
N_CLASSES = 10
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 15000
IS_PRETRAIN = True


def train():

    # pre_trained_weights = './VGG16_pretrain/vgg16.npy'
    data_dir = '/home/yuxin/data/cifar10_data/'
    train_log_dir = './logs2/train/'
    val_log_dir = './logs2/val/'

    with tf.name_scope('input'):
        train_image_batch, train_label_batch = input_data.read_cifar10(data_dir,
                                                                       is_train=True,
                                                                       batch_size=BATCH_SIZE,
                                                                       shuffle=True)

        val_image_batch, val_label_batch = input_data.read_cifar10(data_dir,
                                                                   is_train=False,
                                                                   batch_size=BATCH_SIZE,
                                                                   shuffle=False)

    logits = VGG.VGG16(train_image_batch, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, train_label_batch)
    accuracy = tools.accuracy(logits, train_label_batch)
    my_global_step = tf.Variable(0, trainable=False, name='global_step')
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_H, IMG_W, 3])
    y_ = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, N_CLASSES])

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # load pretrain weights
    # tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            train_images, train_labels = sess.run([train_image_batch, train_label_batch])
            _, train_loss, train_accuracy = sess.run([train_op, loss, accuracy],
                                                     feed_dict={x: train_images, y_: train_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print("Step: %d, loss: %.4f, accuracy: %.4f%%" % (step, train_loss, train_accuracy))
                summary_str = sess.run(summary_op)
                train_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_accuracy = sess.run([loss, accuracy],
                                                  feed_dict={x: val_images, y_: val_labels})
                print("** Step: %d, loss: %.4f, accuracy: %.4f%%" % (step, val_loss, val_accuracy))
                summary_str = sess.run(summary_op)
                val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, save_path=checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limited reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


# evaluate accuracy in test data set
def evaluate():
    with tf.Graph().as_default():

        log_dir = './logs2/train/'
        test_dir = '/home/yuxin/data/cifar10_data/'
        n_test = 10000

        test_iamge_batch, test_label_batch = input_data.read_cifar10(test_dir,
                                                                     is_train=False,
                                                                     batch_size=BATCH_SIZE,
                                                                     shuffle=False)

        logits = VGG.VGG16(test_iamge_batch, N_CLASSES, IS_PRETRAIN)
        correct = tools.num_correct_prediction(logits, test_label_batch)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print('Reading checkpoint...')
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Load success, global step: %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                print('\nEvaluating...')
                num_step = int(math.ceil(n_test / BATCH_SIZE))
                num_example = num_step * BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1

                print("Total test examples: %d" % num_example)
                print("Total correct predictions: %d" % total_correct)
                print("Average accuracy: %.2f%%" % (100 * total_correct / num_example))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
