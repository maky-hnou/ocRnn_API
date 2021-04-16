import os
import yaml
import time
import tensorflow as tf

from tensorflow import keras
from ocRnn.core.decoders import CTCGreedyDecoder


class CharRecognizer():
    def __init__(self, image_model):
        self.image_model = image_model
        self.image = image_model.image
        self.model_path = 'ocRnn/core/model/saved_model.h5'
        self.config_path = 'ocRnn/core/model/config.yml'

    def process_image(self, image_path, shape):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=shape[2])
        if shape[1] is None:
            img_shape = tf.shape(img)
            scale_factor = shape[0] / img_shape[0]
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
            img_width = tf.cast(img_width, tf.int32)
        else:
            img_width = shape[1]
        img = tf.image.resize(img, (shape[0], img_width))
        return img

    def load_configs(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']
        return config

    def run(self):
        model = keras.models.load_model(self.model_path, compile=False)
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        tf.compat.v1.keras.backend.set_session(
            tf.compat.v1.Session(config=tf_config))
        config = self.load_configs(self.config_path)
        decoder = CTCGreedyDecoder(config['table_path'])
        start = time.time()
        # Set the path of the uploaded file
        image_path = os.path.abspath(self.image.url).replace("/media/",
                                                             "media/")
        img = self.process_image(str(image_path), config['img_shape'])
        img = tf.expand_dims(img, 0)
        outputs = model(img)
        processing_time = round(time.time() - start, 2)
        if not isinstance(outputs, tuple):
            outputs = decoder(outputs)
        self.image_model.text = outputs[0].numpy()
        self.image_model.processed = True
        self.image_model.processing_time = processing_time
        self.image_model.save()
