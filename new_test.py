# coding=utf-8

from easydict import EasyDict as edict
import tensorflow as tf
from core import batcher
import functools
from builder import input_reader_builder

import matplotlib.pyplot as plt
import numpy as np

slim = tf.contrib.slim


train_config = edict()
train_config.batch_size = 5
train_config.batch_queue_capacity = 200
train_config.num_batch_queue_threads = 1
train_config.prefetch_queue_capacity = 10

input_reader_config = edict()
input_reader_config.input_path = r'D:\tf_test\tfrecord\train\train.tfrecord-*'
input_reader_config.num_epochs = None
input_reader_config.num_readers = 1
input_reader_config.shuffle = False
input_reader_config.queue_capacity = 256
input_reader_config.min_after_dequeue = 128


def create_input_queue(batch_size, create_tensor_dict_fn,
                       batch_queue_capacity, num_batch_queue_threads,
                       prefetch_queue_capacity):

    tensor_dict = create_tensor_dict_fn()
    tensor_dict['image'] = tf.expand_dims(tensor_dict['image'], 0)
    images = tensor_dict['image']
    float_images = tf.to_float(images)
    tensor_dict['image'] = float_images

    input_queue = batcher.BatchQueue(
        tensor_dict,
        batch_size=batch_size,
        batch_queue_capacity=batch_queue_capacity,
        num_batch_queue_threads=num_batch_queue_threads,
        prefetch_queue_capacity=prefetch_queue_capacity)
    return input_queue


def main():
    create_tensor_dict_fn = functools.partial(input_reader_builder.build, input_reader_config)
    input_queue = create_input_queue(
        train_config.batch_size,
        create_tensor_dict_fn,
        train_config.batch_queue_capacity,
        train_config.num_batch_queue_threads,
        train_config.prefetch_queue_capacity,
    )

    with tf.Session() as sess:
        ini_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess.run(ini_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        for i in range(5):
            tensor_list = input_queue.dequeue()
            single_tensor = tensor_list[0]
            bbox = single_tensor['bbox'] - [1, 1, 2, 2]
            print('bbox = ', sess.run(bbox))



if __name__ == '__main__':
    main()

