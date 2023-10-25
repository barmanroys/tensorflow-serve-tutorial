#!/usr/bin/env python3
# encoding: utf-8
"""The main file to run."""

import logging
import os
import random
from typing import Any, Dict, List, Text, Tuple

import apache_beam as beam
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tfx
from tfx.components.base import base_component, executor_spec, base_executor
from tfx.components.example_gen.base_example_gen_executor import \
    BaseExampleGenExecutor
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.orchestration.experimental.interactive.interactive_context import \
    InteractiveContext
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import channel_utils, channel, standard_artifacts, \
    artifact_utils
from tfx.types.component_spec import ChannelParameter, ComponentSpec, \
    ExecutionParameter

from helpers import get_image_data

logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)


class CustomIngestionComponentSpec(ComponentSpec):
    """ComponentSpec for Custom Ingestion Component."""

    PARAMETERS = {
        'name': ExecutionParameter(type=Text),
        'input_base': ExecutionParameter(type=Text),
    }
    INPUTS = {}
    OUTPUTS = {
        'examples': ChannelParameter(type=standard_artifacts.Examples),
    }


class CustomIngestionExecutor(base_executor.BaseExecutor):
    """Executor for CustomIngestionComponent."""

    # noinspection PyPep8Naming,PyUnusedLocal
    @staticmethod
    def Do(input_dict: Dict[Text, List[standard_artifacts.Artifact]],
           output_dict: Dict[Text, List[standard_artifacts.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        """
        Run the executor. The name and signature follow
        from the base class name.
        """

        input_base: str = exec_properties['input_base']
        image_files: List[str] = tf.io.gfile.listdir(path=input_base)
        random.shuffle(image_files)

        train_images, eval_images = image_files[100:], image_files[:100]
        splits: List[Tuple[str, List[str]]] = [('train', train_images),
                                               ('eval', eval_images)]

        examples_artifact = output_dict['examples'][0]
        examples_artifact.split_names = artifact_utils.encode_split_names(
            ["train", "eval"])

        for split_name, images in splits:

            output_dir: str = artifact_utils.get_split_uri(
                output_dict['examples'], split_name)
            logging.debug(msg=f'The output directory is {output_dir}')
            tf.io.gfile.mkdir(path=output_dir)
            tfrecords_filename = os.path.join(output_dir, 'images.tfrecords')
            with tf.io.TFRecordWriter(path=tfrecords_filename,
                                      options=tf.io.TFRecordOptions(
                                      )) as writer:
                for image_filename in images:
                    image_path = os.path.join(input_base, image_filename)
                    example = get_image_data(image_path)
                    writer.write(record=example.SerializeToString())


class CustomIngestionComponent(base_component.BaseComponent):
    """CustomIngestion Component."""
    SPEC_CLASS = CustomIngestionComponentSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(CustomIngestionExecutor)

    def __init__(self,
                 output_data: channel.Channel = None,
                 input_base: Text = None,
                 name: Text = None):
        if not output_data:
            examples_artifact = standard_artifacts.Examples()
            output_data = channel_utils.as_channel([examples_artifact])

        spec = CustomIngestionComponentSpec(input_base=input_base,
                                            examples=output_data,
                                            name=name)

        super().__init__(spec=spec)


test_context = InteractiveContext()

data_root = os.path.join('PetImages', 'Dog')

ingest_images = CustomIngestionComponent(
    input_base=data_root,
    name='ImageIngestionComponent')
test_context.run(component=ingest_images)

statistics_gen = tfx.components.StatisticsGen(
    examples=ingest_images.outputs['examples'])
test_context.run(component=statistics_gen)

test_context.show(item=statistics_gen.outputs['statistics'])


@beam.ptransform_fn
def image_to_example(
        pipeline: beam.Pipeline,
        input_dict: Dict[Text, List],
        exec_properties: Dict[Text, Any]) -> beam.pvalue.PCollection:
    """Read jpeg files and transform to TF examples.

    Note that each input split will be transformed by this function separately.

    Args:
        pipeline: beam pipeline.
        input_dict: Input dict from input key to a list of Artifacts.
          - input_base: input dir that contains the image data.
        exec_properties: A dict of execution properties.

    Returns:
        PCollection of TF examples.
    """
    # noinspection PyTypeChecker
    image_pattern: str = os.path.join(input_dict['input_base'],
                                      exec_properties)
    logging.info(msg=
    'Processing input image data {} to TFExample.'.format(
        image_pattern))

    image_files = tf.io.gfile.glob(image_pattern)
    if not image_files:
        raise RuntimeError(
            'Split pattern {} does not match any files.'.format(image_pattern))

    # noinspection PyUnresolvedReferences
    return (
            pipeline
            | beam.Create(image_files)
            | 'ConvertImagesToBase64' >> beam.Map(
        lambda file: get_image_data(file))
    )


class ImageExampleGenExecutor(BaseExampleGenExecutor):
    """TFX example gen executor for processing jpeg format.

    Example usage:

      from tfx.components.example_gen.component import FileBasedExampleGen


      example_gen = FileBasedExampleGen(
          input_base="/content/PetImages/",
          input_config=input_config,
          output_config=output,
          custom_executor_spec=executor_spec.ExecutorClassSpec(_Executor))
    """

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        """Returns PTransform for image to TF examples."""
        # noinspection PyTypeChecker
        return image_to_example


pipeline_name: str = "dogs_cats_pipeline"

context = InteractiveContext(pipeline_name=pipeline_name)

input_config = example_gen_pb2.Input(splits=[
    example_gen_pb2.Input.Split(name='images', pattern='*/*.jpg'),
])

output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
        example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
    ]))

example_gen = FileBasedExampleGen(
    input_base="/content/PetImages/",
    input_config=input_config,
    output_config=output,
    custom_executor_spec=executor_spec.ExecutorClassSpec(
        ImageExampleGenExecutor))
context.run(component=example_gen)
statistics_gen = tfx.components.StatisticsGen(
    examples=example_gen.outputs['examples'])
context.run(component=statistics_gen)
context.show(item=statistics_gen.outputs['statistics'])
schema_gen = tfx.components.SchemaGen(
    statistics=statistics_gen.outputs['statistics'])

context.run(component=schema_gen)
context.show(item=schema_gen.outputs['schema'])
example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])
context.run(component=example_validator)
transform = tfx.components.Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath("transform.py"))
context.run(component=transform)

trainer = tfx.components.Trainer(
    module_file=os.path.abspath(path='trainer.py'),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=160),
    eval_args=trainer_pb2.EvalArgs(num_steps=200))
context.run(trainer)
model_resolver = tfx.dsl.Resolver(
    strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
    model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
    model_blessing=tfx.dsl.Channel(
        type=tfx.types.standard_artifacts.ModelBlessing),
)
context.run(component=model_resolver)
eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(label_key='label')
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='AUC'),
            ]
        )
    ],
    slicing_specs=[
        tfma.SlicingSpec()
    ])

evaluator = tfx.components.Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)
context.run(component=evaluator)
context.show(evaluator.outputs['evaluation'])

_serving_model_dir = "/content/exported_model"

pusher = tfx.components.Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=_serving_model_dir)))

context.run(component=pusher)
components = [
    example_gen,
    statistics_gen,
    schema_gen,
    example_validator,
    transform,
    trainer,
    model_resolver,
    evaluator,
    pusher,
]

_pipeline_name = "dogs_cats_pipeline"

# pipeline inputs
_base_dir: str = os.getcwd()
_pipeline_dir: str = os.path.join(_base_dir, "pipeline")

# pipeline outputs
_output_base: str = os.path.join(_pipeline_dir, "output", _pipeline_name)
_pipeline_root: str = os.path.join(_output_base, "pipeline_root")
_metadata_path: str = os.path.join(_pipeline_root, "metadata.sqlite")
