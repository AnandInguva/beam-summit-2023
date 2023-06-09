import argparse
import logging

import apache_beam as beam
from apache_beam.transforms.periodicsequence import PeriodicImpulse

"""
python auto_model_refresh/image_name_to_pubsub.py --runner=DataflowRunner --project google.com:clouddfe --temp_location gs://clouddfe-anandinguva/tmp2 --region us-central1 --streaming
"""

def run(parser):

  known_args, pipeline_args = parser.parse_known_args()
  pipeline  = beam.Pipeline(argv=pipeline_args)
  impulse = (
      pipeline
      | 'CreateImpulse' >> PeriodicImpulse(fire_interval=known_args.interval)
      | 'EmitImageName' >> beam.Map(
    lambda _: known_args.image_name.encode('utf-8'))
  )

  impulse | beam.io.WriteToPubSub(known_args.topic)

  result = pipeline.run().wait_until_finish(5)

  return result
  

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--topic', 
                      default='projects/google.com:clouddfe/topics/anandinguva-ml-updates',
                      help='PubSub topic that is used to emit image name.')
  parser.add_argument('--image_name',
                      default='gs://anandinguva-test/auto_model_updates/dog_image.png',
                      help='Image name to emit.')
  parser.add_argument('--interval',
                      default=20,
                      help='Interval in seconds to emit image name.')
  run(parser)