import tensorflow as tf
from Preprocessing.preprocessing import read_image_test
from Data_Generator.generator import Datagenerator


class Predictor:
    def __init__(self, config, test_path):
        self.config = config
        self.graph_path = self.config['network']['graph_path']
        self.model_weight = self.config['train']['output_weight']
        self.x_size = config['dataset']['size_x']
        self.y_size = config['dataset']['size_y']
        self.model = self.load_model()
        self.test_loader = Datagenerator(self.config, test_path, shuffle=False, is_train=False)

    def load_model(self):
        json_file = open(self.graph_path, 'r')
        load_json = json_file.read()
        json_file.close()

        model = tf.keras.models.model_from_json(load_json)
        model.load_weights(self.model_weight)
        return model

    def predict(self):
        class_predict = []
        predictions = self.model.predict(self.test_loader, batch_size=None)
        for prediction in predictions:
            if prediction >= 0.5:
                class_predict.append('Dog')
            else:
                class_predict.append('Cat')
        return class_predict

