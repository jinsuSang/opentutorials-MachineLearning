import tensorflow as tf
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

irisCSV = '../source/iris.csv'
iris = pd.read_csv(irisCSV)
print(iris.head())

iris = pd.get_dummies(iris)
print(iris.head())

independent = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dependent = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]

independent = np.array(independent)
dependent = np.array(dependent)

X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy',
              metrics=[tf.keras.metrics.Accuracy()])

model.fit(independent, dependent, epochs=3000, verbose=0)
model.fit(independent, dependent, epochs=10)

print(model.predict(independent[:5]))
print(dependent[:5])

print(model.predict(independent[-5:]))
print(dependent[-5:])

print(model.get_weights())
print(model.summary())
