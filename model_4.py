'''
  Author       : Bao Jiarong
  Creation Date: 2020-06-09
  email        : bao.salirong@gmail.com
  Task         : Classification using Tensorflow 2
  Dataset      : flowers (daisy,dandelion,rose,sunflower,tulip)
'''

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import cv2
import loader1

# np.random.seed(7)
# tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# Input/Ouptut Parameters
width      = 100
height     = 100
channel    = 3
n_outputs  = 5
model_name = "models/flowers/flower"
data_path  = "data/flowers/train/"

# Step 0: Global Parameters
epochs     = 20
lr_rate    = 0.001
batch_size = 256
initializer= ["glorot_uniform"  , "glorot_normal",
              "he_normal"       , "he_uniform",
              "lecun_uniform"   , "lecun_normal",
              "random_uniform"  , "random_normal",
              "truncated_normal", "zeros"]

w_init = initializer[5] # Fixed
b_init = initializer[-1]

activations= ["tanh","sigmoid","softplus","softsign","relu",tf.nn.relu6,"elu",tf.nn.leaky_relu,"linear"]
layer_act  = [activations[4],activations[0],activations[5],activations[-2],activations[-1]]
sizes = [32,64,128,100,n_outputs]

# Step 1: Create Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(height, width, channel)))

model.add(tf.keras.layers.Conv2D(filters=sizes[0], kernel_size=(3,3),strides=(1,1),
                                 padding           = 'valid',
                                 activation        = layer_act[0],
                                 kernel_initializer= w_init,
                                 bias_initializer  = b_init))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(filters=sizes[1], kernel_size=(3,3),strides=(1,1),
                                 padding           = 'valid',
                                 activation        = layer_act[1],
                                 kernel_initializer= w_init,
                                 bias_initializer  = b_init))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(filters=sizes[2], kernel_size=(3,3),strides=(1,1),
                                 padding           = 'valid',
                                 activation        = layer_act[2],
                                 kernel_initializer= w_init,
                                 bias_initializer  = b_init))
model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units              = sizes[3],
                                activation         = layer_act[3],
                                kernel_initializer = w_init,
                                bias_initializer   = b_init))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units              = sizes[-1],
                                activation         = layer_act[4],
                                kernel_initializer = w_init,
                                bias_initializer   = b_init))
model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.2))

# Step 2: Define Metrics
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = lr_rate),
              loss     = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics  = ['accuracy'])
# print(model.summary())
# sys.exit()

if sys.argv[1] == "train":
    # Step 3: Load data
    X_train, Y_train, X_test, Y_test = loader1.load_data(data_path,width,height)

    # Step 4: Training
    # Create a function that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_name,
                                                     save_weights_only=True,
                                                     verbose=0, save_freq="epoch")
    # model.load_weights(model_name)
    model.fit(X_train, Y_train,
              batch_size     = batch_size,
              epochs         = epochs,
              validation_data= (X_test,Y_test),
              callbacks      = [cp_callback])

    # Step 6: Evaluation
    loss,acc = model.evaluate(X_test, Y_test, verbose = 2)
    print("Evaluation, accuracy: {:5.2f}%".format(100 * acc))

elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    img = cv2.imread(sys.argv[2])
    image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
    images = np.array([image])
    images = loader1.scaling_tech(images,method="normalization")

    # Step 5: Predict the
    preds = my_model.predict(images)
    t = np.argmax(preds[0])
    kinds_name = ["daisy","dandelion","rose","sunflower","tulip"]

    for i in range(n_outputs):
        if np.argmax(preds[0]) == i:
            print(kinds_name[i])
