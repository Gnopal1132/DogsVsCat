import tensorflow as tf
from functools import partial

class Models:
    def __init__(self, config):
        self.config = config
        self.x_size = config['dataset']['size_x']
        self.y_size = config['dataset']['size_y']
        self.channels = config['dataset']['channel']
        self.model_img = config['network']['model_img']
        self.learning_rate = config['train']['learning_rate']
        self.optimizer = config['train']['optimizer']

    def convolution_scratch(self, save_model=False):
        tf.keras.backend.clear_session()
        Default = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1, padding='same', activation='relu')
        model = tf.keras.models.Sequential([
            Default(filters=32, kernel_size=7, strides=2, input_shape=[self.x_size, self.y_size, self.channels]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=2),

            Default(filters=64),
            Default(filters=64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.MaxPool2D(pool_size=2),

            Default(filters=128),
            Default(filters=128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.MaxPool2D(pool_size=2),

            Default(filters=256),
            Default(filters=256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.MaxPool2D(pool_size=2),

            Default(filters=512),
            Default(filters=512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.MaxPool2D(pool_size=2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(300, activation='relu', use_bias=False),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        if save_model:
            tf.keras.utils.plot_model(model, show_shapes=True, show_dtype=True, show_layer_names=True, to_file=self.model_img)

        if self.optimizer == 'ADAM':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999)
        elif self.optimizer == 'NADAM':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999)
        elif self.optimizer == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer == 'SGD_MOMENTUM':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        elif self.optimizer == 'RMS_PROP':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, rho=0.9)
        elif self.optimizer == 'ADA_GRAD':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        else:
            raise Exception('Optimizer properly not defined in Configuration!!')

        model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])
        return model

