#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ex1data1.csv")
data.plot.scatter('population','profit')
X = data['population']
y = data['profit']
theta = np.zeros(2)
alpha = 0.01
num_iters = 1500

def predict(X, theta):
    return theta[0] + theta[1] * X

def cost(X, y, theta):
    return np.sum((predict(X, theta) - y) ** 2)

def fit(X, y, theta, alpha, num_iters):
    m = X.size
    for i in range(num_iters):
        tmp = np.zeros(2)
        p = predict(X, theta);
        tmp[0] = - np.sum(p - y) * alpha / m
        tmp[1] = - np.sum((p - y) * X) * alpha / m
        theta = theta + tmp
    return theta

def fit_with_cost(X, y, theta, alpha, num_iters):
    m = X.size
    J_history = np.empty(num_iters)
    for i in range(num_iters):
        J_history[i] = cost(X, y, theta)
        tmp = np.zeros(2)
        p = predict(X, theta);
        tmp[0] = - np.sum(p - y) * alpha / m
        tmp[1] = - np.sum((p - y) * X) * alpha / m
        theta = theta + tmp
    return theta, J_history

def visualize(theta):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlim([4.5,22.5])
    ax.set_ylim([-5, 25])
    ax.scatter(X, y)
    line_x = np.linspace(0,22.5, 20)
    line_y = theta[0] + line_x * theta[1]
    ax.plot(line_x, line_y)
    plt.show()

theta, J_history = fit_with_cost(X, y, theta, alpha, num_iters)
visualize(theta)

fig = plt.figure()
ax = plt.axes()
ax.plot(J_history)
        
