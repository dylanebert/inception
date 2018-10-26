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

def append_last_layer(base_model, n_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
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

def predict():
    datagen = ImageDataGenerator(rescale=1/255.)
    train_gen = datagen.flow_from_directory('/data/nlp/gmc/train', target_size=(299, 299), batch_size=100, shuffle=False)
    predict_gen = datagen.flow_from_directory('/data/nlp/vae/model/gmc/images/original', target_size=(299, 299), batch_size=100, shuffle=False)

    n_classes = len(train_gen.class_indices)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = append_last_layer(base_model, n_classes)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('model/gmc/best.ckpt')

    index_class_dict = {k: v for v, k in train_gen.class_indices.items()}
    filenames = predict_gen.filenames
    n = len(predict_gen)
    for i in range(n):
        images, labels = predict_gen[i]
        predictions = model.predict(images, verbose=1)
        for j in range(len(images)):
            top_prediction_labels = [index_class_dict[item] for item in np.argsort(predictions[j])[::-1]][:100]
            print(filenames[100 * i + j])
            print(top_prediction_labels)

    '''index_class_dict = {k: v for v, k in train_gen.class_indices.items()}
    filenames = gen.filenames
    n = len(gen)
    for i in range(n):
        print('{0} of {1}'.format(i+1, n), end='\r')
        images, labels = gen[i]
        predictions = model.predict(images, verbose=0)
        for j in range(len(images)):
            correct_label = index_class_dict[np.argmax(labels[j])]
            top_prediction_labels = [index_class_dict[item] for item in np.argsort(predictions[j])[::-1]][:100]
            line = json.dumps({'label': str(correct_label), 'filename': filenames[batch_size * i + j], 'predictions': top_prediction_labels})
            with open(os.path.join(predictions_path, correct_label + '.json'), 'a+') as f:
                f.write('{0}\n'.format(line))'''

if __name__ == '__main__':
    predict()
