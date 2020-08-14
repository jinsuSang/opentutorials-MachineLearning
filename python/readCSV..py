import pandas as pd

bostonCSV = "../source/bonston.csv"
boston = pd.read_csv(bostonCSV)

irisCSV = "../source/iris.csv"
iris = pd.read_csv(irisCSV)

lemonadeCSV = "../source/lemonade.csv"
lemonade = pd.read_csv(lemonadeCSV)

print(boston.shape)
print(iris.shape)
print(lemonade.shape)

print(lemonade.columns)
print(boston.columns)
print(iris.columns)

independent = lemonade[['온도']]
dependent = lemonade[['판매량']]
print(independent.shape, dependent.shape)

independent = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dependent = iris[['품종']]
print(independent.shape, dependent.shape)

print(lemonade.head())
print(boston.head())
print(iris.head())
