import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("datasets/column_2C_weka.csv")

# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data['class'] == 'Abnormal']  # Filtering for abnormal class
x = np.array(data1.loc[:, ['pelvic_incidence', "lumbar_lordosis_angle"]]).reshape(-1, 2)  # Take pelvic_incidence colum as features
y = np.array(data1.loc[:, 'sacral_slope']).reshape(-1, 1)  # Take sacral_slope colum as labels

# Split train, test part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# LinearRegression
multiple_linear_regression = LinearRegression()
# Predict space
# Fit
multiple_linear_regression.fit(x_train, y_train)
# Predict
predicted = multiple_linear_regression.predict(x_test)
print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2: ", multiple_linear_regression.coef_)

# predict
y_pred = multiple_linear_regression.predict(x_test)

print("r_square score", r2_score(y_test, y_pred))
