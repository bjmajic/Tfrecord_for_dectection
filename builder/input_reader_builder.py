# coding=utf-8
"""
input reader builder.
Create data sources for DetectionModels from an InputReader config
Note: If users wishes to also use their own InputReaders with Object Detection configuration
framework, they should define their own builder function that wraps the build function
"""
import tensorflow as tf

from data_decoders import  tf_example_decoder

parallel_reader = tf.contrib.slim.parallel_reader


def build(input_reader_config):
    """
    Builds a tensor dictionary based on the InputReader config
    :param input_reader_config:
    :return:  A tensor dict based on the input_reader_config.
    """
    _, string_tensor = parallel_reader.parallel_read(data_sources=input_reader_config.input_path,
                                                     reader_class=tf.TFRecordReader,
                                                     num_epochs=input_reader_config.num_epochs,
                                                     num_readers=input_reader_config.num_readers,
                                                     shuffle=input_reader_config.shuffle,
                                                     dtypes=[tf.string, tf.string],
                                                     capacity=input_reader_config.queue_capacity,
                                                     min_after_dequeue=input_reader_config.min_after_dequeue)
    decoder = tf_example_decoder.TfExampleDecoder()
    return decoder.decode(string_tensor)
