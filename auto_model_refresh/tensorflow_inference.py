
"""
1. Source - Read image name from PubSub
2. Do some preprocessing
3. RunInference
4. Visualize the results.


Side inputs: 
1. When a model is updated to GCS, update the model
"""

import argparse
import tensorflow as tf

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.ml.inference import RunInference
from apache_beam.ml.inference.tensorflow_inference import TFModelHandlerTensor
from apache_beam.ml.inference.base import PredictionResult
import numpy as np
from PIL import Image


def preprocess_image(image_name, image_dir):
  img = tf.keras.utils.get_file(image_name, image_dir + image_name)
  img = Image.open(img).resize((224, 224))
  img = np.array(img) / 255.0
  img_tensor = tf.cast(tf.convert_to_tensor(img[...]), dtype=tf.float32)
  return img_tensor

def post_process_result(result: PredictionResult):
  # Do some post processing
  return result

def run(parser):
  known_args, pipeline_args = parser.parse_known_args()
  options = PipelineOptions(pipeline_args)
  tf_model_handler = TFModelHandlerTensor(model_uri=known_args.model_path)
  with beam.Pipeline(options=options) as p:
    file_pattern = 'gs://apache-beam-ml/side_inputs/resnet50*'
    
    with beam.Pipeline(options=options) as p:
      image_name_from_pubsub = (
        p
        | "ReadImageNameFromPubSub" >> beam.io.ReadFromPubSub(
            known_args.input_topic)
        | "DecodeImageName" >> beam.Map(lambda x: x.decode('utf-8'))
      )

      prediction_result = (
        image_name_from_pubsub
        | "RunInference" >> RunInference(
          model_handler=tf_model_handler,
          file_pattern=file_pattern
        ).with_preprocess_fn(preprocess_image
        ).with_postprocess_fn(post_process_result)
      )

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_topic',
                      # please replace the default topic with your own topic.
                      default='projects/google.com:clouddfe/topics/anandinguva-ml-updates',
                      help='PubSub topic that emits image name.')
  parser.add_argument('--model_path',
                      required=True,
                      help='Path to the TensorFlow model.'
                      'This wil be used to load the initial model.')
  run(parser)