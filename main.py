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
    gen = datagen.flow_from_directory(train_dir, target_size=(image_size, image_size), batch_size=batch_size, shuffle=False)

    n_classes = len(gen.class_indices)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = append_last_layer(base_model, n_classes)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights(weights_path)

    index_class_dict = {k: v for v, k in gen.class_indices.items()}
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
                f.write('{0}\n'.format(line))

def membership():
    datagen = ImageDataGenerator(rescale=1/255.)
    gen = datagen.flow_from_directory(train_dir, target_size=(image_size, image_size), batch_size=batch_size, shuffle=False)

    n_classes = len(gen.class_indices)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = append_last_layer(base_model, n_classes)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights(weights_path)

    index_class_dict = {k: v for v, k in gen.class_indices.items()}
    filenames = gen.filenames
    n = len(gen)
    for i in range(n):
        print('{0} of {1}'.format(i+1, n), end='\r')
        images, labels = gen[i]
        predictions = model.predict(images, verbose=0)
        for j in range(len(images)):
            top_prediction_labels = [index_class_dict[item] for item in np.argsort(predictions[j])[::-1]][:50]
            for word in gen.class_indices.keys():
                prediction_val = predictions[j][gen.class_indices[word]]
                for r in [1, 5, 10, 25, 50]:
                    if word in top_prediction_labels[:r]:
                        with open(os.path.join(membership_path, 'r' + str(r), word), 'a+') as f:
                            f.write('{0}\n'.format(filenames[batch_size * i + j]))
                for p in [.5, .75, .9]:
                    if prediction_val >= p:
                        with open(os.path.join(membership_path, ('p' + str(p)).replace('.','').replace('0',''), word), 'a+') as f:
                            f.write('{0}\n'.format(filenames[batch_size * i + j]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='data directory, containing train/dev/test folders', type=str, required=True)
    parser.add_argument('--model_path', help='directory in which to store', type=str, required=True)
    parser.add_argument('--hidden_size', help='appended hidden layer size', type=int, default=300)
    parser.add_argument('--train', help='train for given max epochs', type=int, default=0)
    parser.add_argument('--predict', help='generate predictions for given data', action='store_true')
    parser.add_argument('--membership', help='determine class membership for each train example', action='store_true')
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
    membership_path = os.path.join(args.model_path, 'membership')
    for membership_type in ['r1', 'r5', 'r10', 'r25', 'r50', 'p5', 'p75', 'p9']:
        if not os.path.exists(os.path.join(membership_path, membership_type)):
            os.makedirs(os.path.join(membership_path, membership_type))

    if args.train is not 0:
        train()
    if args.predict:
        predict()
    if args.membership:
        membership()
