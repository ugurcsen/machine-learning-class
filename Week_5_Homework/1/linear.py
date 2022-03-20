import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("datasets/column_2C_weka.csv")

# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data['class'] == 'Abnormal']  # Filtering for abnormal class
x = np.array(data1.loc[:, 'pelvic_incidence']).reshape(-1, 1)  # Take pelvic_incidence colum as features
y = np.array(data1.loc[:, 'sacral_slope']).reshape(-1, 1)  # Take sacral_slope colum as labels
# Scatter
plt.figure(figsize=[10, 10])
plt.scatter(x=x, y=y)  # Scatter graph created
plt.xlabel('pelvic_incidence')  # x axis label set
plt.ylabel('sacral_slope')  # y axis label set
plt.show()

# LinearRegression
reg = LinearRegression()
# Predict space
# Create an array between min(x) and max(x) which length 50 and step sizes same
predict_space = np.linspace(min(x), max(x)).reshape(-1, 1)
# Fit
reg.fit(x, y)
# Predict
predicted = reg.predict(predict_space)
# R^2
print('R^2 score: ', reg.score(x, y))
# Plot regression line and scatter
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x, y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
