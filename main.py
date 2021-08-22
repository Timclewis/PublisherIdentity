import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, Xception, EfficientNetB3, EfficientNetB4, EfficientNetB5
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from PIL import ImageFile

from DataGenerator import DataGenerator
from BuildModel import BuildModel
from Logger import Logger
from RunModel import run_model

# import Predictor

ImageFile.LOAD_TRUNCATED_IMAGES = True
print(f'Tensorflow version {tf.version.VERSION}')
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Args (self, val_split, height, width, batch_size, steps):
dat1 = DataGenerator(val_split=0.15, height=300, width=200, batch_size=16, steps=100)
dat1.generator_info()

mod1 = BuildModel(data_generator=dat1)
mod1.compile_model(epochs=1, network=Xception, pooling='avg', optimizer=Adam, learn_rate=1E-4, summary=False)
log1 = Logger(build_model=mod1, sprite_height=100, sprite_width=70, data_points=500)
#
# Args (self, epochs, summary, metrics, confusion_matrix, image_visual, projector):
run_model(build_model=mod1, logger=log1, metrics=True, confusion_matrix=True, image_visual=True, projector=True)
#
# pred1 = Predictor.Predictor(mod1)
# pred1.img_predict()
# pred1.batch_predict(1000)
# pred1.batch_evaluate(49)
