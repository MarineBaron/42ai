#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ex2data1.csv")

X = data[['exam1', 'exam2']]
X.insert(0, 'const', 1, 1)
X = np.array(X)
y = data['admission']

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))
	
def predict(X, theta):
    return sigmoid(X.dot(theta))
	
def cost(X, y, theta):
    h = predict(X, theta);
    return ((-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) / len(y))
	
def fit(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        p = predict(X, theta);
        theta -=  X.T.dot(p - y) * alpha / m
        J_history.append(cost(X, y, theta))
    return theta, J_history
	
def exactitude(p, y):
	return np.sum([1 if ((y[i] == 0 and p[i] < 0.5) or (y[i] == 1 and p[i] >= 0.5)) else 0 for i in range(len(p))])

theta, J_history = fit(X, y, theta, 0.001, 10000)
p = predict(X, theta)
e = exactitude(p, y)

print(e)

fig = plt.figure()
ax = plt.axes()
colors = ['red' if i==0 else 'blue'for i in data['admission']]
ax.set_xlim([np.min(data['exam1']), np.max(data['exam1'])])
ax.set_ylim([np.min(data['exam2']), np.max(data['exam2'])])
ax.scatter(data['exam1'], data['exam2'], 20, colors, 'o')
plt.show()
