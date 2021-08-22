import io
import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Utils:
    # Code from Tensorflow tutorial Tensorboard
    @staticmethod
    def plot_to_image(figure):
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        # Use white text if squares are dark; otherwise black.
        threshold = max(cm) / 2
        figure = plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Greens'), vmin=0, vmax=1)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    @staticmethod
    def get_img(img_path):
        # img = Image.open(StringIO(img_path))
        img = tf.io.read_file(img_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # resize the image to the desired size for your model
        img = tf.image.resize(img, (200, 300))
        return img
