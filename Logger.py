import os
import random as r
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import confusion_matrix
from tensorboard.plugins import projector
from tensorflow.keras.models import Model, load_model
from os.path import join, exists
from Utils import Utils


class Logger:
    def __init__(self, build_model, sprite_height, sprite_width, data_points):
        self.width = build_model.width
        self.height = build_model.height
        self.test_data = build_model.test_data
        self.class_names = build_model.class_names
        self.class_num = build_model.class_num
        self.test_num = build_model.test_data.n
        self.test_labels = build_model.test_labels

        self.network = build_model.network
        self.pool = build_model.pool
        self.opt = build_model.opt
        self.lr = build_model.lr
        self.epochs = build_model.epochs
        self.batch_size = build_model.batch_size
        self.current_time = build_model.current_time

        self.model = build_model.model
        self.embeddings = Model(inputs=build_model.model.inputs,
                                outputs=build_model.model.layers[-2].output)

        self.sprite_width = sprite_width
        self.sprite_height = sprite_height
        self.datapoint = data_points

        self.log_dir = join(f'notebooks/{self.network.__name__}{self.pool}_'
                                    f'{self.opt.__name__}lr{self.lr}_E{self.epochs}'
                                    f'B{self.batch_size}-{self.current_time}')
        self.file_writer_cm = tf.summary.create_file_writer(f'{self.log_dir}/cm')
        self.file_writer_image = tf.summary.create_file_writer(f'{self.log_dir}/image')

    def log_projection(self, epoch):
        log_dir = join('tensorboard_logs')
        if not exists(log_dir):
            os.makedirs(log_dir)

        # Generate embeddings
        images_pil = []
        images_embeddings = []
        labels = []

        for x in list(r.sample(range(0, self.test_num), self.datapoint)):
            img_path = self.test_data.filepaths[x]
            img_tf = Utils.get_img(img_path)

            # Save both tf image for prediction and PIL image for sprite
            img_pil = np.array(Image.open(img_path).resize((self.sprite_width, self.sprite_height)))
            img_embedding = self.embeddings(tf.expand_dims(img_tf, axis=0))
            images_embeddings.append(np.array(img_embedding[0]))
            images_pil.append(img_pil)

            # create and store labels
            label = img_path[77:-4]
            labels.append(label)
            with open(join(f'{log_dir}/projector/', 'metadata.tsv'), 'w') as f:
                for label in labels:
                    f.write(f'{label}\n')

        one_square_size = int(np.ceil(np.sqrt(len(images_embeddings))))
        tile_width = self.sprite_width * one_square_size
        tile_height = self.sprite_height * one_square_size
        sprite_image = Image.new(mode='RGBA', size=(tile_width, tile_height), color=(0, 0, 0, 0))

        for count, image in enumerate(images_pil):
            div, mod = divmod(count, one_square_size)
            h_loc = self.sprite_height * div
            w_loc = self.sprite_width * mod
            image = Image.fromarray(image)
            sprite_image.paste(image, (w_loc, h_loc, w_loc + self.sprite_width, h_loc + self.sprite_height))
        sprite_image.save(join(f'{log_dir}/projector/', 'sprite.png'))

        feature_vector = tf.Variable(images_embeddings)
        checkpoint = tf.train.Checkpoint(embedding=feature_vector)
        checkpoint.save(join(f'{log_dir}/projector/', 'embedding.ckpt'))

        # Set up config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
        embedding.metadata_path = 'projector/metadata.tsv'
        embedding.sprite.image_path = 'projector/sprite.png'
        embedding.sprite.single_image_dim.extend((self.sprite_width, self.sprite_height))
        projector.visualize_embeddings(log_dir, config)

    def log_confusion_matrix(self, epoch):
        # Use the model to predict the values from the test_images.
        test_pred_raw = self.model.predict(self.test_data)
        test_pred = np.argmax(test_pred_raw, axis=1)
        logs: print(self.batch_size)

        # Calculate the confusion matrix using sklearn.metrics
        cm = confusion_matrix(self.test_labels, test_pred)
        figure = Utils.plot_confusion_matrix(cm, class_names=self.class_names)
        cm_image = Utils.plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image(f'{self.network.__name__}-{self.current_time}', cm_image, step=epoch)

    def log_images(self, epoch):
        # Create pubs list
        pubs = list(r.sample(range(0, self.test_num), pow(self.class_num, 2)))
        img_data = []
        for i in pubs:
            img = load_img(self.test_data.filepaths[i])
            img = img.resize((self.width, self.height))  # width x height
            img_arr = np.asarray(img)
            img_data.append(img_arr)

        # Data should be in (BATCH_SIZE, H, W, C)
        assert np.size(np.shape(img_data)) == 4
        # Create a figure to contain the plot.
        figure = plt.figure(figsize=(10, 10))
        num_images = np.shape(img_data)[0]
        size = int(np.ceil(np.sqrt(num_images)))

        for i in range(len(pubs)):
            # Start next subplot.
            plt.subplot(size, size, i + 1, title=self.class_names[self.test_labels[pubs[i]]])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img_data[i], cmap=plt.get_cmap('Binary'))

        with self.file_writer_image.as_default():
            tf.summary.image(f'{self.network.__name__}-{self.current_time}',
                             Utils.plot_to_image(figure), max_outputs=len(pubs), step=epoch)
