import imghdr
import os
import cv2

from collections import Counter
from PIL import ImageFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

PATH_TRAIN = 'C:/Users/crims/PycharmProjects/publisherIdentify/data/training/'
PATH_TEST = 'C:/Users/crims/PycharmProjects/publisherIdentify/data/testing/'


class DataGenerator:
    def __init__(self, val_split, height, width, batch_size, steps):
        self.val_split = val_split
        self.val_steps = steps * val_split * 2
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.steps = steps

        train_generator = ImageDataGenerator(rescale=1. / 255, validation_split=self.val_split)
        print('Training folder:')
        self.train_data = train_generator.flow_from_directory(PATH_TRAIN, target_size=(self.height, self.width),
                                                              class_mode='categorical', batch_size=self.batch_size,
                                                              subset='training')
        print('Validation folder:')
        self.valid_data = train_generator.flow_from_directory(PATH_TRAIN, target_size=(self.height, self.width),
                                                              class_mode='categorical', batch_size=self.batch_size,
                                                              subset='validation')

        test_generator = ImageDataGenerator(rescale=1. / 255)
        print('Test folder:')
        self.test_data = test_generator.flow_from_directory(PATH_TEST, target_size=(self.width, self.height),
                                                            class_mode=None, batch_size=1, shuffle=False)
        print()
        self.class_num = self.train_data.num_classes
        self.class_counter = list(Counter(self.train_data.classes).values())
        self.class_counter_valid = list(Counter(self.valid_data.classes).values())
        self.class_names = list(self.train_data.class_indices)
        self.labels = self.train_data.labels
        self.test_labels = list(range(self.class_num)) * (int(self.test_data.n) // self.class_num)

    def generator_info(self):
        # Print info about the generated data
        print(f'Train data class name and num {dict(zip(self.class_names, self.class_counter))}')
        print(f'Valid data class name and num {dict(zip(self.class_names, self.class_counter_valid))}')
        print(
            f'Files trained {self.batch_size * self.steps} '
            f'and validated {(self.batch_size * self.val_steps):.0f} per epoch')
        print(
            f'Images resized to {self.height}x{self.width} '
            f'and all files trained in {(self.train_data.n // (self.batch_size * self.steps)):.2f} epochs')
        print()

    def clean_data(self, folder):
        if folder:
            data = self.test_data
        else:
            data = self.train_data
        # Remove nontype files from data folder (run once EVER for data)
        filenames = data.filenames
        n = 0
        while n < data.n:
            path = f'{PATH_TRAIN}{filenames[n]}'
            image = cv2.imread(path)
            img_type = imghdr.what(path)
            if img_type != "jpeg":
                print(f'Removing image from {path}')
                os.remove(path)
                n += 1
            else:
                n += 1
        print('All done!')
