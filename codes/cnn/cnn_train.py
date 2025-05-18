print("Starting...")

import os
import numpy as np
import random
from tqdm import tqdm
from glob import glob
from keras.utils import to_categorical
from keras.preprocessing import image
from sklearn.datasets import load_files

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
import tensorflow as tf

# Paths
training_path = r'C:\Users\admin\Desktop\L.D.D\data\badam\testing'
testing_path = r'C:\Users\admin\Desktop\L.D.D\data\badam\training'
save_dir = r'C:\Users\admin\Desktop\L.D.D\codes\cnn\savedModels'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("Done importing")

# Load dataset
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = to_categorical(np.array(data['target']), 3)  # 3 classes
    return files, targets

print("Loading data...")

train_files, train_targets = load_dataset(training_path)
test_files, test_targets = load_dataset(testing_path)

print("Done loading data")

random.seed(9)

def path_to_tensor(img_path):
    try:
        img = image.load_img(img_path, target_size=(64, 64))
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)
    except:
        return np.zeros((1, 64, 64, 3))

def paths_to_tensor(img_paths):
    tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(tensors)

# Preprocessing
print("Preprocessing data...")
train_tensors = paths_to_tensor(train_files).astype('float32') / 255
test_tensors = paths_to_tensor(test_files).astype('float32') / 255

# Model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))  # 3 classes

model.summary()

# Compile and Train
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training...")
model.fit(train_tensors, train_targets, epochs=4, batch_size=64,
          validation_data=(test_tensors, test_targets), verbose=1)

# Evaluate
loss, acc = model.evaluate(test_tensors, test_targets)
print('Test Loss:', loss)
print('Test Accuracy:', acc)

# === Save models ===

# Save as .keras
keras_model_path = os.path.join(save_dir, "leaf_badam_model.keras")
model.save(keras_model_path)
print(f"✅ Keras model saved at: {keras_model_path}")