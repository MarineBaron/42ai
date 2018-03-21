#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ex1data2.csv")
data.plot.scatter('nb_bedrooms','price')
plt.show()
