import numpy as np
import os
import tensorflow as tf
import keras
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='data directory, containg train/dev/test folders', type=str, required=True)
parser.add_argument('--model_path', help='directory in which to store', type=str, required=True)
parser.add_argument('--hidden_size', help='appended hidden layer size', type=int, default=300)
parser.add_argument('--train', help='train for given max epochs', type=int, default=0)
parser.add_argument('--predict', help='predict each label', action='store_true')
args = parser.parse_args()

image_size = 299
batch_size = 100
hidden_size = args.hidden_size
n_freeze = 172
train_dir = os.path.join(args.data_path, 'train')
dev_dir = os.path.join(args.data_path, 'dev')
test_dir = os.path.join(args.data_path, 'test')
weights_path = os.path.join(args.model_path, 'best.ckpt')
overfit_path = os.path.join(args.model_path, 'overfit.ckpt')
predictions_path = os.path.join(args.model_path, 'predictions')

def append_last_layer(base_model, n_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(hidden_size, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    return model

def transfer_learning(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['categorical_accuracy'])

def fine_tuning(model):
    for layer in model.layers[:n_freeze]:
        layer.trainable = False
    for layer in model.layers[n_freeze:]:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=.0001), loss='binary_crossentropy', metrics=['categorical_accuracy'])

def train():
    datagen = ImageDataGenerator(rescale=1/255.)
    train_generator = datagen.flow_from_directory(train_dir, target_size=(image_size, image_size), batch_size=batch_size, shuffle=False)
    dev_generator = datagen.flow_from_directory(dev_dir, target_size=(image_size, image_size), batch_size=batch_size, shuffle=False)

    n_classes = len(train_generator.class_indices)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = append_last_layer(base_model, n_classes)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(weights_path, save_best_only=True, verbose=1)
    earlystopping_callback = keras.callbacks.EarlyStopping(verbose=1, patience=2)

    transfer_learning(model, base_model)
    history_tl = model.fit_generator(train_generator, validation_data=dev_generator, epochs=1000, callbacks=[checkpoint_callback, earlystopping_callback])

    fine_tuning(model)
    model.load_weights(weights_path)
    history_ft = model.fit_generator(train_generator, validation_data=dev_generator, epochs=1000, callbacks=[checkpoint_callback, earlystopping_callback])

    model.save_weights(overfit_path)

def predict():
    datagen = ImageDataGenerator(rescale=1/255.)
    test_generator = datagen.flow_from_directory(test_dir, target_size=(image_size, image_size), batch_size=batch_size, shuffle=False)

    n_classes = len(test_generator.class_indices)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = append_last_layer(base_model, n_classes)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights(weights_path)

    index_class_dict = {k: v for v, k in test_generator.class_indices.items()}
    filenames = test_generator.filenames
    n = len(test_generator)
    for i in range(n):
        print('{0} of {1}'.format(i+1, n), end='\r')
        images, labels = test_generator[i]
        predictions = model.predict(images, verbose=0)
        for j in range(len(labels)):
            correct = index_class_dict[np.argmax(labels[j])]
            preds = [index_class_dict[item] for item in np.argsort(predictions[j])[::-1]][:100]
            pred_val = predictions[j][np.argmax(labels[j])]
            line = json.dumps({'label': str(correct), 'filename': filenames[batch_size * i + j], 'p': float(pred_val), 'predictions': preds})
            with open(os.path.join(predictions_path, correct + '.json'), 'a+') as f:
                f.write('{0}\n'.format(line))

if __name__ == '__main__':
    if args.train is not 0:
        train()
    if args.predict:
        predict()
