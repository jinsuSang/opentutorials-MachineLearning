import tensorflow as tf
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

irisCSV = '../source/iris.csv'
iris = pd.read_csv(irisCSV)
iris = pd.get_dummies(iris)


independent = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dependent = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]

independent = np.array(independent)
dependent = np.array(dependent)

X = tf.keras.layers.Input(shape=[4])

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

Y = tf.keras.layers.Dense(3, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)

model.compile(loss='categorical_crossentropy',
              metrics=[tf.keras.metrics.Accuracy()])
print(model.summary())

model.fit(independent, dependent, epochs=3000, verbose=0)
model.fit(independent, dependent, epochs=10)

print(model.predict(independent[:5]))
print(dependent[:5])

print(model.predict(independent[-5:]))
print(dependent[-5:])
