import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("datasets/column_2C_weka.csv")

# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data['class'] == 'Abnormal']  # Filtering for abnormal class
x = np.array(data1.loc[:, 'pelvic_incidence']).reshape(-1, 1)  # Take pelvic_incidence colum as features
y = np.array(data1.loc[:, 'sacral_slope']).reshape(-1, 1)  # Take sacral_slope colum as labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

polynomial_regression = PolynomialFeatures(degree=2)

x_train_polynomial = polynomial_regression.fit_transform(x_train)
x_test_polynomial = polynomial_regression.fit_transform(x_test)

# %% fit
p_regression = LinearRegression()
p_regression.fit(x_train_polynomial, y_train)

# Testing our results

y_pred = p_regression.predict(x_test_polynomial)

plt.scatter(x, y)
plt.xlabel("engine")
plt.ylabel("car_price")
plt.plot(x_test, y_pred, color="green", label="poly")
plt.legend()
plt.show()
# Calculating r2 score / Approximation of regression line

print("r_square score for polynomial regression: ", r2_score(y_test, y_pred))
