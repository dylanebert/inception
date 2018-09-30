import numpy as np
import os
import tensorflow as tf
import keras
import json
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from refit import append_last_layer

image_size = 299
batch_size = 100
hidden_size = 1024
test_dir = '/data/gmc/test'

datagen = ImageDataGenerator(rescale=1/255.)
test_generator = datagen.flow_from_directory(test_dir, target_size=(image_size, image_size), batch_size=batch_size)

n_classes = len(test_generator.class_indices)

base_model = InceptionV3(weights='imagenet', include_top=False)
model = append_last_layer(base_model, n_classes)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('model/best.ckpt')

index_class_dict = {k: v for v, k in test_generator.class_indices.items()}
for i in range(len(test_generator)):
    images, labels = test_generator[i]
    predictions = model.predict(images, verbose=1)
    for j in range(len(labels)):
        correct = index_class_dict[np.argmax(labels[j])]
        preds = [index_class_dict[item] for item in np.argsort(predictions[j])[::-1]][:100]
        pred_val = predictions[j][np.argmax(labels[j])]
        line = json.dumps({'label': str(correct), 'p': float(pred_val), 'predictions': preds})
        with open(os.path.join('predictions', correct + '.json'), 'a+') as f:
            f.write('{0}\n'.format(line))
