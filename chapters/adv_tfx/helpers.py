#!/usr/bin/env python3
# encoding: utf-8
"""Provide some helper functions for feature engineering."""

import tensorflow as tf


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_label_from_filename(filename: str) -> int:
    """ Function to set the label for each image. In our case, we'll use the
    file
    path of a label indicator. Based on your initial data
    Args:
      filename: string, full file path
    Returns:
      0 for dog, 1 for cat
    Raises:
      NotImplementedError if not label category was detected

    """

    lowered_filename: str = filename.lower()
    if "dog" in lowered_filename:
        return 0
    if "cat" in lowered_filename:
        return 1
    raise NotImplementedError


def _convert_to_example(image_buffer: bytes, label: int) -> tf.train.Example:
    """Function to convert image byte strings and labels into tf.Example
    structures
      Args:
        image_buffer: byte string representing the image
        label: int
      Returns:
        TFExample data structure containing the image (byte string) and the
        label (int encoded)
    """

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/raw': _bytes_feature(image_buffer),
                'label': _int64_feature(label)
            }))
    return example


def get_image_data(filename: str) -> tf.train.Example:
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
    Returns:
      TFExample data structure containing the image (byte string) and the
      label (int encoded)
    """
    label: int = get_label_from_filename(filename)
    byte_content: tf.Tensor = tf.io.read_file(filename=filename)
    return _convert_to_example(byte_content.numpy(), label)
