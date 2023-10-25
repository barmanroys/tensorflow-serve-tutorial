#!/usr/bin/env python3
# encoding: utf-8
"""Provide some helper transformation functions."""

from typing import Union, Dict

import numpy as np
import tensorflow as tf

import constants


@tf.function
def convert_image(raw_image: tf.Tensor) -> tf.Tensor:
    """Give a converted image for jpg, else give a constant tensor of zeros."""
    if tf.io.is_jpeg(contents=raw_image):
        image = tf.io.decode_jpeg(raw_image, channels=3)
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        image = tf.image.resize(image,
                                [constants.IMG_SIZE, constants.IMG_SIZE])
    else:
        image = tf.constant(
            value=np.zeros(shape=(constants.IMG_SIZE, constants.IMG_SIZE, 3)),
            dtype=tf.float32)
    return image


def fill_in_missing(x: Union[tf.Tensor, tf.SparseTensor]) -> tf.Tensor:
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a dense
    tensor.

    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at
      most 1
        in the second dimension.

    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = "" if x.dtype == tf.string else 0
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value,
        )
    return tf.squeeze(x, axis=1)


def preprocessing_fn(inputs: Dict[str, Union[tf.Tensor, tf.SparseTensor]]) -> \
        Dict[str, tf.Tensor]:
    """tf.transform's callback function for preprocessing inputs.
    """
    outputs = {}

    for key in constants.RAW_FEATURE_KEYS:
        image = fill_in_missing(inputs[key])
        outputs[constants.transformed_name(key)] = tf.map_fn(convert_image,
                                                             image,
                                                             dtype=tf.float32)

    outputs[constants.transformed_name(constants.LABEL_KEY)] = inputs[
        constants.LABEL_KEY]

    return outputs
