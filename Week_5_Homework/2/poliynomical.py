import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

data = pd.read_csv("car_details_model_ready.csv")
data = data.drop(
    ['Unnamed: 0', 'name', 'seller_type', 'fuel', 'transmission', 'torque'], axis=1)
imputer = SimpleImputer(strategy="median")  #
imputer.fit(data)
data2 = imputer.transform(data)  # Fill missing values
data = pd.DataFrame(data2, columns=data.columns)

y = data["selling_price"].values.reshape(-1, 1)
x = data.drop(["selling_price"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

polynomial_regression = PolynomialFeatures(degree=2)

x_train_polynomial = polynomial_regression.fit_transform(x_train)
x_test_polynomial = polynomial_regression.fit_transform(x_test)

# %% fit
p_regression = LinearRegression()
p_regression.fit(x_train_polynomial, y_train)

# Testing our results

y_pred = p_regression.predict(x_test_polynomial)

plt.scatter(x.engine, y)
plt.xlabel("engine")
plt.ylabel("car_price")
plt.plot(x_test.engine, y_pred, color="green", label="poly")
plt.legend()
plt.show()
# Calculating r2 score / Approximation of regression line

print("r_square score for polynomial regression: ", r2_score(y_test, y_pred))
