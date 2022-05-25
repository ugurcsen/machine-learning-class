import pandas as pd
import seaborn as sns

# read csv (comma separated value) into data
from matplotlib import pyplot as plt

data = pd.read_csv('datasets/column_2C_weka.csv')
# to see features and target variable
print(data.head().to_string())
# Well know question is there any NaN value and length of this data so lets look at info
print(data.info())

print(data.describe().to_string())

color_list = ['red' if i == 'Abnormal' else 'green' for i in data.loc[:, 'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                           c=color_list,
                           figsize=[15, 15],
                           diagonal='hist',
                           alpha=0.5,
                           s=200,
                           marker='*',
                           edgecolor="black")
plt.show()

sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()
