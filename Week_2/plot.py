import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Read data from pokemon.csv
data = pd.read_csv('./data/pokemon.csv')

# Print head of Pokemon dataset
# print(data.head().to_string())

# Print info of Pokemon dataset
# print(data.info())

# Print Correlations of Pokemon dataset
# print(data.corr().to_string())

# Correlation map of Pokemon dataset
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# Print data columns
# print(data.columns)

# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind='line', color='g', label='Speed', linewidth=1, alpha=0.5, grid=True, linestyle=':')
data.Defense.plot(color='r', label='Defense', linewidth=1, alpha=0.5, grid=True, linestyle='-.')
plt.legend(loc='upper right')  # legend = puts label into plot
plt.xlabel('x axis')  # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')  # title = title of plot
plt.show()

# Scatter Plot
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color='red')
plt.xlabel('Attack')  # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')  # title = title of plot
plt.show()

# Histogram
# bins = number of bar in figure
data.Speed.plot(kind='hist', bins=50, figsize=(12, 12))
plt.show()

# clf() = cleans it up again you can start a fresh
data.Speed.plot(kind='hist', bins=50)
plt.clf()
# We can not see plot due to clf()
