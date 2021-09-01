import numpy as np
import tensorflow as tf
import os

K = tf.keras.backend


class one_cycle(tf.keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None, last_iteration=None, last_rate=None):
        super().__init__()
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iteration = last_iteration or iterations // 10 + 1
        self.half_iteration = (self.iterations - self.last_iteration) // 2
        self.last_rate = last_rate or max_rate / 1000
        self.iteration = 0
        self.loss = []
        self.rate = []

    def __interpolate(self, start_iteration, final_iteration, start_rate, final_rate):
        return ((final_rate - start_rate)*(self.iteration - start_iteration))/((final_iteration-start_iteration) + start_rate)

    def on_batch_begin(self, batch, logs=None):
        if self.iteration < self.half_iteration:
            rate = self.__interpolate(0, self.half_iteration,self.start_rate,self.max_rate)
        elif self.iteration < 2*self.half_iteration:
            rate = self.__interpolate(self.half_iteration, 2*self.half_iteration, self.max_rate, self.start_rate)
        else:
            rate = self.__interpolate(2*self.half_iteration, self.iterations,self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)

    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs['loss'])
        self.rate.append(K.get_value(self.model.optimizer.lr))


class exponential_scheduler(tf.keras.callbacks.Callback):
    def __init__(self, s=40000):
        super().__init__()
        self.s = s
        self.loss = []
        self.rate = []

    def on_batch_begin(self, batch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, lr*0.001*(1/self.s))

    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs['loss'])
        self.rate.append(K.get_value(self.model.optimizer.lr))


class Result_callback(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_epoch = []
        self.val_loss_epoch = []
        self.rate = []
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        self.rate.append(K.get_value(self.model.optimizer.lr))
        self.loss_epoch.append(logs['loss'])
        self.val_loss_epoch.append((logs['val_loss']))
        self.epoch.append(epoch)


class Trainer:
    def __init__(self, config, trainsize, model, train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.batch_size = self.config['train']['batch_size']
        self.epochs = self.config['train']['epochs']
        self.train_size = trainsize
        self.results = Result_callback()
        self.callbacks = self.get_callbacks().append(self.results)

    def get_id(self, root_dir):
        import time
        id_ = time.strftime('run_id_%Y_%m_%D_%H_%M_%S')
        return os.path.join(root_dir, id_)

    def get_callbacks(self):

        callbacks = []
        if self.config['callbacks']['earlystopping']['use_early_stop']:
            patience = self.config['callbacks']['earlystopping']['patience']
            monitor = self.config['callbacks']['earlystopping']['monitor']
            early_stop = tf.keras.callbacks.EarlyStopping(patience=patience, monitor=monitor)
            callbacks.append(early_stop)

        if self.config['callbacks']['checkpoint']['checkpoint_last']['enabled']:
            monitor = self.config['callbacks']['checkpoint']['checkpoint_last']['monitor']
            file_path = self.config['callbacks']['checkpoint']['checkpoint_last']['out_last']
            checkpoint_last = tf.keras.callbacks.ModelCheckpoint(monitor=monitor,
                                                                 save_best_only=False,
                                                                 save_weights_only=True,
                                                                 filepath=file_path)
            callbacks.append(checkpoint_last)

        if self.config['callbacks']['checkpoint']['checkpoint_best']['enabled']:
            monitor = self.config['callbacks']['checkpoint']['checkpoint_best']['monitor']
            file_path = self.config['callbacks']['checkpoint']['checkpoint_best']['out_last']
            checkpoint_best = tf.keras.callbacks.ModelCheckpoint(monitor=monitor,
                                                                 save_best_only=True,
                                                                 save_weights_only=True,
                                                                 filepath=file_path)
            callbacks.append(checkpoint_best)

        if self.config['callbacks']['tensorboard']['enabled']:
            directory = self.get_id(self.config['callbacks']['tensorboard']['log_dir'])
            board = tf.keras.callbacks.TensorBoard(directory, write_graph=True, write_images=True)
            callbacks.append(board)

        if self.config['callbacks']['scheduler']['onecycle']['to_use']:
            iterations = np.ceil(self.train_size/self.batch_size) * self.epochs
            max_rate = self.config['callbacks']['scheduler']['onecycle']['max_rate']
            one_cycle_callback = one_cycle(iterations=iterations, max_rate=max_rate)
            callbacks.append(one_cycle_callback)

        if self.config['callbacks']['scheduler']['exponential_scheduler']['to_use']:
            s = self.config['callbacks']['scheduler']['exponential_scheduler']['params']
            exponential_scheduler_callback = exponential_scheduler(s=s)
            callbacks.append(exponential_scheduler_callback)
        return callbacks

    def train(self):
        # Lets now train this bad boy

        if self.config['train']['weight_initialization']['use_pretrained']:
            reload_from = self.config['train']['weight_initialization']['restore_from']
            print(f"Restoring model weights from: {reload_from}")
            self.model.load_weights(reload_from)
        else:
            print(f"Saving graph in: {self.config['network']['graph_path']}")
            self.save_graph(self.model, self.config['network']['graph_path'])

        use_multiprocessing = self.config["train"]['use_multiprocessing']
        num_workers = self.config["train"]["num_workers"]

        self.model.fit_generator(generator=self.train_loader, validation_data=self.val_loader,
                                 callbacks=self.callbacks, epochs=self.epochs, use_multiprocessing=use_multiprocessing,
                                 workers=num_workers, shuffle=False, max_queue_size=10, verbose=1)

        print(f"Saving Weights in {self.config['train']['output_weight']}")
        self.model.save_weights(self.config['train']['output_weight'])

    def save_graph(self, model, path):
        model_json = model.to_json()
        with open(path, "w") as json_file:
            json_file.write(model_json)


