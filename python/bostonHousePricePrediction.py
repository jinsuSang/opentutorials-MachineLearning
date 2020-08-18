import tensorflow as tf
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

bostonPriceCSV = '../source/boston.csv'
bostonPrice = pd.read_csv(bostonPriceCSV)
print(bostonPrice.columns)
bostonPrice.head()

independent = bostonPrice[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
                           'ptratio', 'b', 'lstat']]
dependent = bostonPrice[['medv']]

independent = np.array(independent)
dependent = np.array(dependent)

print(independent.shape, dependent.shape)

X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)

model = tf.keras.models.Model(X, Y)
model.compile(loss="mse")

model.fit(independent, dependent, epochs=1600, verbose=0)
model.fit(independent, dependent, epochs=10)

print(model.predict(independent[5:10]))
print(dependent[5:10])

print(model.get_weights())
print(model.summary())
