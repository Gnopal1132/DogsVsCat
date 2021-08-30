import sys
import os
import tensorflow as tf
import numpy as np
from Preprocessing.preprocessing import read_image,read_image_test
#sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))


class Datagenerator(tf.keras.utils.Sequence):
    def __init__(self, config, dataset, shuffle=True, is_train=True):
        self.config = config
        self.dataset = dataset
        self.is_train = is_train
        self.len_dataset = len(dataset)
        self.indices = np.arange(self.len_dataset)
        self.shuffle = shuffle
        self.classes = config['dataset']['classes']
        self.aug = config['data_aug']['use_aug']
        self.x_size = config['dataset']['size_x']
        self.y_size = config['dataset']['size_y']
        self.channels = config['dataset']['channel']
        self.batch_size = config['train']['batch_size']

        if self.shuffle:
            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):   # Returns number of batches per epoch
        return int(np.floor(self.len_dataset/self.batch_size))

    def __getitem__(self, index):  # Generated batch for given index

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        id = [self.dataset[k] for k in indices]
        if self.is_train:
            x, y = self.__data_generation(id)
            return x, y
        else:
            x = self.__data_generation(id)
            return x

    def __data_generation(self, ids):

        x_batch = []
        y_batch = []
        if self.is_train:
            for instance in ids:
                image, label = read_image(instance, size=(self.x_size, self.y_size), to_aug=self.aug)
                x_batch.append(image)
                y_batch.append(label)
            x_batch = np.asarray(x_batch, dtype=np.float32)
            y_batch = np.asarray(y_batch, dtype=np.float32)
            return x_batch, y_batch
        else:
            batch = []
            for img in ids:
                image = read_image_test(img, size=(self.x_size, self.y_size))
                batch.append(image)
            return np.asarray(batch)
