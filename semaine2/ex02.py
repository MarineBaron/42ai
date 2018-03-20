#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

data = pd.read_csv("ex1data2.csv")
X = data[['size', 'nb_bedrooms']]
X.insert(0, 'const', 1, 1)
y = data['price']
theta = np.zeros(3)
theta.fill(1)
alpha = 0.01
num_iters = 10000

def predict(X, theta):
    return X.dot(theta)

def featureNormalize(X):
    mean = np.zeros(3)
    stdev = np.zeros(3)
    j = 0;
    for i in X:
        mean[j] = np.mean(X[i])
        stdev[j] = np.std(X[i])
        if (stdev[j] != 0):
            X[i] = (X[i] - mean[j]) / stdev[j]
        j += 1
        
    return X, mean, stdev

X, mean, stdev = featureNormalize(X)

def cost(X, y, theta):
    return np.sum((predict(X, theta) - y) ** 2)

def fit_with_cost(X, y, theta, alpha, num_iters):
    m = X.size
    J_history = np.empty(num_iters)
    for i in range(num_iters):
        J_history[i] = cost(X, y, theta)
        tmp = np.zeros(3)
        p = predict(X, theta);
        tmp[0] = - np.sum(p - y) * alpha / m
        tmp[1] = - np.sum((p - y) * X['size']) * alpha / m
        tmp[2] = - np.sum((p - y) * X['nb_bedrooms']) * alpha / m
        theta = theta + tmp
    return theta, J_history
    
theta, J_history = fit_with_cost(X, y, theta, alpha, num_iters)
print(theta)

fig = plt.figure()
ax = plt.axes()
ax.plot(J_history)

values = data[['size', 'nb_bedrooms', 'price']]
for i in range(values['size'].size - 2):
    price = theta[0] + (values['size'][i] - mean[1]) / stdev[1] * theta[1] + (values['nb_bedrooms'][i] - mean[2]) / stdev[2] * theta[2]
    print(price, " ", values['price'][i], ' ', price - values['price'][i])


tmp = np.zeros(3);
tmp[0] = theta[0] - mean[1] * theta[1] / stdev[1] - mean[2] * theta[2] / stdev[2]
tmp[1] = theta[1] / stdev[1]
tmp[2] = theta[2] / stdev[2]
print(' ')
for i in range(values['size'].size - 2):
    price = tmp[0] + values['size'][i]  * tmp[1] + values['nb_bedrooms'][i] * tmp[2]
    print(price, " ", values['price'][i], ' ', price - values['price'][i])

print(tmp)

X = data[['size', 'nb_bedrooms']]   
X['b'] = 1 
from statsmodels.api import OLS
model = OLS(y, X)
result = model.fit()
print(result.summary())
print('Parameters: ', result.params)

for i in range(values['size'].size - 2):
    price = result.params['b'] + values['size'][i] * result.params['size'] + values['nb_bedrooms'][i] * result.params['nb_bedrooms']
    print(price, " ", values['price'][i], ' ', price - values['price'][i])
    

  