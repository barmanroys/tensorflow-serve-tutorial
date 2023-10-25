#!/usr/bin/env python3
# encoding: utf-8
"""Provide some helper method for training."""

from typing import List, Text, Tuple

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.executor import TrainerFnArgs

import constants

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_label_for_image(model, tf_transform_output):
    """Returns a function that parses a raw byte image and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_images_fn(image_raw):
        """Returns the output to be used in the serving signature."""

        image_raw = tf.reshape(image_raw, [-1, 1])
        parsed_features = {'image': image_raw}
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_images_fn


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies
    TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""

        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(constants.LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples,
                                              feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 32) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
      file_pattern: input tfrecord file pattern.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=constants.transformed_name(constants.LABEL_KEY))

    return dataset


def get_model() -> tf.keras.Model:
    """Creates a CNN Keras model based on transfer learning for classifying
    image data.

    Returns:
      A keras Model.
    """
    img_shape: Tuple = (constants.IMG_SIZE, constants.IMG_SIZE, 3)
    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                   include_top=False)
    base_model.trainable = False
    model = tf.keras.Sequential(layers=[
        tf.keras.layers.Input(shape=img_shape, name=constants.transformed_name(
            constants.INPUT_KEY)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=1)])
    model.compile(optimizer=tf.optimizers.RMSprop(lr=0.01),
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.metrics.BinaryAccuracy(name='accuracy')])
    return model


def run_fn(fn_args: TrainerFnArgs):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output,
                              TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output,
                             EVAL_BATCH_SIZE)

    model = get_model()

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
    )

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                      tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),

    }
    model.save(fn_args.serving_model_dir, save_format='tf',
               signatures=signatures)
