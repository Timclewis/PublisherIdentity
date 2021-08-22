import random as r
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import BuildModel
import DataGenerator


class Predictor(BuildModel):
    def __init__(self, build_model):
        self.test_data = build_model.test_data
        self.valid_data = build_model.valid_data
        self.test_labels = build_model.test_labels
        self.width = build_model.width
        self.height = build_model.height
        self.model = build_model.model
        self.class_names = build_model.class_names
        self.batch_size = build_model.batch_size

    def img_predict(self):
        n = r.randint(0, self.test_data.n)
        filenames = self.valid_data.filenames
        path = f'{DataGenerator.PATH_TEST}{filenames[n]}'
        pic = mpimg.imread(path)
        plt.axis('off')
        plt.imshow(pic)
        plt.show()

        img = tf.keras.preprocessing.image.load_img(path, target_size=(self.width, self.height))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_processed = tf.keras.applications.xception.preprocess_input(img_batch)

        prediction = self.model(img_processed, training=False)
        top_index = np.argsort(np.max(prediction, axis=0))[-1]
        second_index = np.argsort(np.max(prediction, axis=0))[-2]

        sort = np.sort(max(prediction))
        print(f'Prediction {self.class_names[top_index]} with conf {round(sort[len(sort) - 1] * 100)}%')
        print(f'2nd predict {self.class_names[second_index]} with conf {round(sort[len(sort) - 2] * 100)}%')
        print(f'Answer is {filenames[n][:]}')

    def batch_evaluate(self, num_batches):
        self.steps = num_batches
        acc = self.model.evaluate(self.valid_data, batch_size=self.batch_size, steps=self.steps)
        print(f'Predictions acc on valid_data is {100 * acc:.1f}')

    def batch_predict(self, num_predicts):
        i = num_predicts
        test_pred_raw = self.model.predict(self.test_data)
        test_pred = np.argmax(test_pred_raw, axis=1)
        test_labels = self.test_labels
        acc = sum(1 for x, y in zip(test_pred[0:i], test_labels[0:i]) if x == y) / len(test_labels[0:i])
        print(f'Prediction acc on test_data is {100 * acc:.1f}')
