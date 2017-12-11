# coding=utf-8
import tensorflow as tf

from core import standard_fields as fields
from core import batcher

slim = tf.contrib.slim


def create_input_queue(batch_size, create_tensor_dict_fn, batch_queue_capacity, num_batch_queue_threads,
                       prefetch_queue_capacity, data_augmentation_options):
    """
    Sets up reader, prefetcher and returns input queue
    :param batch_size:
    :param create_tensor_dict_fn: function to create tensor dictionary.
    :param batch_queue_capacity: maximum number of elements to store within a queue.
    :param num_batch_queue_threads: number of threads to use for batching.
    :param prefetch_queue_capacity:  maximum capacity of the queue used to prefetch
                             assembled batches.
    :param data_augmentation_options: a list of tuples, where each tuple contains a
      data augmentation function and a dictionary containing arguments and their
      values (see preprocessor.py).
    :return: input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
      (which hold images, boxes and targets).  To get a batch of tensor_dicts,
      call input_queue.Dequeue().
    """
    tensor_dict = create_tensor_dict_fn()
    tensor_dict[fields.InputDataFields.image] = tf.expand_dims(tensor_dict[fields.InputDataFields.image], 0)
    images = tensor_dict[fields.InputDataFields.image]
    float_images = tf.to_float(images)
    tensor_dict[fields.InputDataFields.image] = float_images

    input_queue = batcher.BatchQueue(
        tensor_dict,
        batch_size=batch_size,
        batch_queue_capacity=batch_queue_capacity,
        num_batch_queue_threads=num_batch_queue_threads,
        prefetch_queue_capacity=prefetch_queue_capacity)
    return input_queue
