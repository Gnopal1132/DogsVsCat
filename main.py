import os
import warnings
from Prediction import predictor
from Models.vision_model import Models
from Data_Generator.generator import Datagenerator
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2
from pathlib import Path
from Trainer import trainer


def read_dataset(source_path, shuffle=True):
    files = os.listdir(source_path)
    preprocessed_paths = []
    for instance in files:
        preprocessed_paths.append(os.path.join(source_path, instance))
    preprocessed_paths = np.asarray(preprocessed_paths)
    if shuffle:
        np.random.shuffle(preprocessed_paths)  # It does inplace
    return preprocessed_paths


def show_some_image_prediction(images, labels, path_to_save=None):

    n_rows = 10
    n_cols = 5

    plt.figure(figsize=(20, 20))
    for row in range(n_rows):
        for col in range(n_cols):
            idx = n_cols * row + col
            plt.subplot(n_rows, n_cols, idx + 1)
            img = cv2.imread(images[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = labels[idx]
            plt.imshow(img)
            plt.title(label, fontsize=12)
            plt.axis('off')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()


def show_some_images(list_images, path_to_save=None):
    n_rows = 5
    n_cols = 5
    plt.figure(figsize=(10, 8))
    for row in range(n_rows):
        for col in range(n_cols):
            idx = n_cols*row + col
            plt.subplot(n_rows, n_cols, idx+1)
            img = cv2.imread(list_images[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = 'Cat' if 'cat' in str(os.path.basename(list_images[idx])) else 'Dog'
            plt.imshow(img)
            plt.title(label, fontsize=12)
            plt.axis('off')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


config_file = Path(os.path.join(os.curdir, "Configuration", "config"))
if not config_file.exists():
    raise Exception("configuration missing!!")
else:
    with open(config_file) as f:
        config_file = yaml.load(f)

train_path = os.path.join(config_file["dataset"]["path"], 'train')
test_path = os.path.join(config_file["dataset"]["path"], 'test1')

data = read_dataset(train_path)
train_size = int(len(data)*config_file["dataset"]["train_size"])
train = data[:train_size]
val = data[train_size:]
test = read_dataset(test_path, shuffle=False)
show_some_images(train, config_file['dataset']['path_image'])  #To see some dataset images

train_loader = Datagenerator(config_file, train, shuffle=True)
val_loader = Datagenerator(config_file, val, shuffle=True)

baseline_model = Models(config=config_file).convolution_scratch(save_model=True)
trainer = trainer.Trainer(config=config_file, model=baseline_model, train_loader=train_loader, val_loader=val_loader)
trainer.train()

# Prediction
predict = predictor.Predictor(config_file, test)
class_predict = predict.predict()
show_some_image_prediction(test, class_predict, path_to_save=config_file['dataset']['predict_image'])

