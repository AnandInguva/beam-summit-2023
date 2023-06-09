
"""
1. Source - Read image name from PubSub
2. Do some preprocessing
3. RunInference
4. Visualize the results.


Side inputs: 
1. When a model is updated to GCS, update the model

python auto_model_refresh/tensorflow_inference.py --runner=DataflowRunner --project google.com:clouddfe --teocation gs://clouddfe-anandinguva/tmp2 --region us-central1 --streaming --requirements_cache=skip --requirements_file=/Users/anandinguva/projects/beam/sdks/python/apache_beam/ml/inference/tensorflow_tests_requirements.txt --num_workers=2
"""

import logging
import argparse
import tensorflow as tf

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.ml.inference import RunInference
from apache_beam.ml.inference.tensorflow_inference import TFModelHandlerTensor
from apache_beam.ml.inference.base import PredictionResult
import numpy as np
from PIL import Image
from apache_beam.io.filesystems import FileSystems
import io
from apache_beam.ml.inference.utils import WatchFilePattern


def read_image(image_file_name):
  with FileSystems().open(image_file_name, 'r') as file:
    data = Image.open(io.BytesIO(file.read())).convert('RGB')
  img = data.resize((224, 224))
  img = np.array(img) / 255.0
  img_tensor = tf.cast(tf.convert_to_tensor(img[...]), dtype=tf.float32)
  return img_tensor

def post_process_result(element: PredictionResult):
  # Do some post processing
    predicted_class = np.argmax(element.inference[0], axis=-1)
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    predicted_class_name = imagenet_labels[predicted_class]
    return predicted_class_name.title(), element.model_id

def run(parser):
  known_args, pipeline_args = parser.parse_known_args()
  options = PipelineOptions(pipeline_args)
  tf_model_handler = TFModelHandlerTensor(model_uri=known_args.model_path)
  with beam.Pipeline(options=options) as p:

    file_pattern = 'gs://anandinguva-test/auto_model_updates/resnet*.h5'

    sideinput = p | "WatchModelPattern" >> WatchFilePattern(
      file_pattern=file_pattern, interval=60)
    
    image_name_from_pubsub = (
      p
      | "ReadImageNameFromPubSub" >> beam.io.ReadFromPubSub(
          known_args.input_topic)
      | "DecodeImageName" >> beam.Map(lambda x: x.decode('utf-8'))
    )

    image_name_from_pubsub |= "PreProcessImage" >> beam.Map(read_image)
    
    prediction_result = (
      image_name_from_pubsub
      | "RunInference" >> RunInference(
        model_handler=tf_model_handler,
        model_metadata_pcoll=sideinput
    ))

    prediction_result |= "PostProcessResult" >> beam.Map(post_process_result)

    prediction_result | "PrintPredictionResult" >> beam.Map(logging.info)

if __name__ == '__main__':
  import logging
  logging.getLogger().setLevel(logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_topic',
                      # please replace the default topic with your own topic.
                      default='projects/google.com:clouddfe/topics/anandinguva-ml-updates',
                      help='PubSub topic that emits image name.')
  parser.add_argument('--model_path',
                      default='gs://anandinguva-test/auto_model_updates/resnet101.h5',
                      help='Path to the TensorFlow model.'
                      'This wil be used to load the initial model.')
  run(parser)