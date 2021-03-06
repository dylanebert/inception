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
import pickle
import h5py
from tqdm import tqdm

def append_last_layer(base_model, n_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(hidden_size, activation='relu')(x)
    predictions = Dense(n_classes, activation='sigmoid')(x)
    model = Model(base_model.input, predictions)
    return model

def transfer_learning(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(lr=.0001), loss='binary_crossentropy', metrics=['top_k_categorical_accuracy'])

def fine_tuning(model):
    for layer in model.layers[:n_freeze]:
        layer.trainable = False
    for layer in model.layers[n_freeze:]:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=.0001), loss='binary_crossentropy', metrics=['top_k_categorical_accuracy'])

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
    gen = datagen.flow_from_directory(train_dir, target_size=(image_size, image_size), batch_size=batch_size, shuffle=False)

    n_classes = len(gen.class_indices)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = append_last_layer(base_model, n_classes)
    model.compile(optimizer=Adam(lr=.0001), loss='binary_crossentropy', metrics=['top_k_categorical_accuracy'])
    model.load_weights(weights_path)

    index_class_dict = {k: v for v, k in gen.class_indices.items()}
    with open(os.path.join(args.model_path, 'index_class_dict.p'), 'wb+') as f:
        pickle.dump(index_class_dict, f)
    filenames = gen.filenames
    n = len(gen)
    current_class_predictions = {}
    prev_label = None
    for i in tqdm(range(n)):
        images, labels = gen[i]
        predictions = model.predict(images, verbose=0)
        for j in range(len(images)):
            correct_label = index_class_dict[np.argmax(labels[j])]
            prediction_vals = list(predictions[j])
            if prev_label is None:
                prev_label = correct_label
            if correct_label is not prev_label:
                path = os.path.join(args.model_path, 'predictions', prev_label + '.p')
                if os.path.exists(path):
                    print('Error: {0} already exists'.format(path))
                else:
                    with open(path, 'wb+') as f:
                        pickle.dump(current_class_predictions, f)
                prev_label = correct_label
                current_class_predictions = {}
            current_class_predictions[filenames[batch_size * i + j]] = prediction_vals

def encode():
    datagen = ImageDataGenerator(rescale=1/255.)
    gen = datagen.flow_from_directory(train_dir, target_size=(image_size, image_size), batch_size=batch_size, shuffle=False)

    n_classes = len(gen.class_indices)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = append_last_layer(base_model, n_classes)
    model.load_weights(weights_path)

    embedding_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    embeddings = embedding_model.predict_generator(gen, verbose=1)

    with h5py.File(os.path.join(args.model_path, 'encodings.hdf5'), 'w') as f:
        f.create_dataset('encodings', data=embeddings)
        f.create_dataset('filenames', data=np.array(gen.filenames, dtype='S'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data directory, containing train/dev/test folders', type=str, required=True)
    parser.add_argument('--model_path', help='directory in which to store data', type=str, required=True)
    parser.add_argument('--hidden_size', help='appended hidden layer size', type=int, default=300)
    parser.add_argument('--train', help='train model', action='store_true')
    parser.add_argument('--predict', help='generate and save all prediction values', action='store_true')
    parser.add_argument('--encode', help='encode data to classifier embeddings', action='store_true')
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

    if args.train:
        train()
    if args.predict:
        predict()
    if args.encode:
        encode()
