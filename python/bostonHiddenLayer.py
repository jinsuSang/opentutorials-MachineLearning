import numpy as np
import pandas as pd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


bostonPriceCSV = '../source/boston.csv'
bostonPrice = pd.read_csv(bostonPriceCSV)

independent = bostonPrice[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
                           'ptratio', 'b', 'lstat']]
dependent = bostonPrice[['medv']]

independent = np.array(independent)
dependent = np.array(dependent)

X = tf.keras.layers.Input(shape=[13])

H = tf.keras.layers.Dense(8)(X)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('selu')(H)

H = tf.keras.layers.Dense(8)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('selu')(H)

H = tf.keras.layers.Dense(8)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('selu')(H)

H = tf.keras.layers.Dense(8)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('selu')(H)

Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss="mse")

print(model.summary())

model.fit(independent, dependent, epochs=1000, verbose=0)
model.fit(independent, dependent, epochs=10)

print(model.predict(independent[:5]))
print(dependent[:5])
