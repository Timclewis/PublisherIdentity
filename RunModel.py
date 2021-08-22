import datetime
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LambdaCallback, TensorBoard, ModelCheckpoint, Callback
from tensorflow.keras.models import Model, load_model

from Logger import Logger


def run_model(build_model, logger, metrics, confusion_matrix, image_visual, projector):
    model = build_model.model
    train_data = build_model.train_data
    valid_data = build_model.valid_data
    epochs = build_model.epochs
    steps = build_model.steps
    val_steps = build_model.val_steps
    batch_size = build_model.batch_size

    log_dir = logger.log_dir

#    checkpoint = ModelCheckpoint(log_dir=f'{log_dir}/checkpoint', monitor='val_acc', save_best_only=True)

    if metrics:
        scalar_callback = TensorBoard(log_dir=log_dir, update_freq=100,
                                      histogram_freq=0, profile_batch=100,
                                      write_graph=True, write_images=True)
    else:
        scalar_callback = LambdaCallback(on_train_begin=print('Skipping metrics'))

    if confusion_matrix:
        cm_callback = LambdaCallback(on_train_begin=print('Skipping confusion matrix'))
        #cm_callback = LambdaCallback(on_epoch_end=Logger.log_confusion_matrix)
    else:
        cm_callback = LambdaCallback(on_train_begin=print('Skipping confusion matrix'))

    if image_visual:
        image_callback = LambdaCallback(on_epoch_end=Logger.log_images)
    else:
        image_callback = LambdaCallback(on_train_begin=print('Skipping image grid'))

    if projector:
        projector_callback = LambdaCallback(on_train_end=Logger.log_projection)
    else:
        projector_callback = LambdaCallback(on_train_begin=print('Skipping 3D projection'))

    model.fit(train_data, validation_data=valid_data,
              callbacks=[scalar_callback, cm_callback, image_callback, projector_callback],
              epochs=epochs, steps_per_epoch=steps,
              validation_steps=val_steps,
              batch_size=batch_size, verbose=1)
