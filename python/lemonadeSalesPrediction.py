import tensorflow as tf
import pandas as pd

lemonadeCSV = '../source/lemonade.csv'
lemonade = pd.read_csv(lemonadeCSV)
print(lemonade.head())

independent = lemonade[['온도']]
dependent = lemonade[['판매량']]

X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

model.fit(independent, dependent, epochs=13000, verbose=0)
model.fit(independent, dependent, epochs=10)

print(model.predict(independent))
print(model.predict([[15]]))
