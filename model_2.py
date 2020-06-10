'''
  Author       : Bao Jiarong
  Creation Date: 2020-06-09
  email        : bao.salirong@gmail.com
  Task         : Classification using Tensorflow 2
  Dataset      : fashion (0,1,...,9)
'''

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import random
import cv2
import loader

# tf.random.set_seed(7)
# np.random.seed(7)
# random.seed(3)
# np.set_printoptions(threshold=np.inf)

# Step 0: Global Parameters
epochs     = 20
lr_rate    = 0.001
batch_size = 32
width      = 28
height     = 28
channel    = 3
model_name = "models/FASHION/fashion"
data_path  = "data/fashion/train/"
n_outputs  = 10

# Step 1: Create Model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(height,width,channel)),
                                    # tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(128,activation='relu'),
                                    tf.keras.layers.Dense(n_outputs,activation='tanh')])

# Step 2: Define Metrics
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = lr_rate),
              loss     = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics  = ['accuracy'])
# print(model.summary())
# sys.exit()

if sys.argv[1] == "train":
    # Step 3: Load data
    X_train, Y_train, X_test, Y_test = loader.load_data(data_path,width,height)

    # Step 4: Training
    # Create a function that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_name,
                                                     save_weights_only=True,
                                                     verbose=0, save_freq=1)
    model.load_weights(model_name)
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
    image = cv2.resize(img,(28,28),interpolation = cv2.INTER_AREA)
    images = np.array([image])
    images = loader.scaling_tech(images,method="normalization")

    # Step 5: Predict the class
    preds = my_model.predict(images)
    t = np.argmax(preds[0])
    kinds_name = ["T-shirt","Trouser","Pullover","Dress","Coat","sandal","shirt","Sneaker","bag","Ankle_boot"]

    for i in range(n_outputs):
        if np.argmax(preds[0]) == i:
            print(kinds_name[i])
