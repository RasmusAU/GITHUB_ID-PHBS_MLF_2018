#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:11:25 2018
@author: Rasmus
"""

## General modules start
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
## General modules end

## Import data start
data = pd.read_csv("/Users/Rasmus/Documents/AU/Udenlandsophold/PKU/Machine Learning for Finance/Exam/Project/IMDb/movie_metadata.csv")
## Import data end
data.dropna(subset=['director_name'], inplace = True)
data = data.iloc[np.r_[0:5, -5:0],1:-2]
print(data)

'''
import matplotlib.pyplot as plt
import numpy as np

# select setosa and versicolor
y = data.iloc[0:2, 3].values
y = np.where(y == 'duration', 180, 185)
plt.scatter(y[:100,0], y[:100,1], color='red', marker='o', label='Title year')

# extract sepal length and petal length
X = data.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
plt.show()

## Merge data start
##env = marketdata_sample + news_sample
## Merge data end

## Data preprocessing start
##env.isnull().sum()
## Data preprocessing end

'''