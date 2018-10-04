import numpy as np
import os
import tensorflow as tf
import keras
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

image_size = 299
batch_size = 100
hidden_size = 1024
n_freeze = 172
train_dir = '/data/zap50k/train'
dev_dir = '/data/zap50k/dev'

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
    train_generator = datagen.flow_from_directory(train_dir, target_size=(image_size, image_size), batch_size=batch_size)
    dev_generator = datagen.flow_from_directory(dev_dir, target_size=(image_size, image_size), batch_size=batch_size)

    n_classes = len(train_generator.class_indices)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = append_last_layer(base_model, n_classes)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='logs/')
    checkpoint_callback = keras.callbacks.ModelCheckpoint('model/best.ckpt', save_best_only=True, verbose=1)
    earlystopping_callback = keras.callbacks.EarlyStopping(verbose=1, patience=2)

    transfer_learning(model, base_model)
    history_tl = model.fit_generator(train_generator, validation_data=dev_generator, epochs=1000, callbacks=[tensorboard_callback, checkpoint_callback, earlystopping_callback])

    fine_tuning(model)
    model.load_weights('model/best.ckpt')
    history_ft = model.fit_generator(train_generator, validation_data=dev_generator, epochs=1000, callbacks=[tensorboard_callback, checkpoint_callback, earlystopping_callback])

    model.save_weights('model/overfit.ckpt')

if __name__ == '__main__':
    train()
