import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("car_details_model_ready.csv")
data = data.drop(
    ['Unnamed: 0', 'name', 'seller_type', 'fuel', 'transmission', 'torque'], axis=1)
imputer = SimpleImputer(strategy="median")  #
imputer.fit(data)
data2 = imputer.transform(data)  # Fill missing values
data = pd.DataFrame(data2, columns=data.columns)

y = data["selling_price"].values.reshape(-1, 1)
x = data.drop(["selling_price"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# %% fitting data
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x_train, y_train)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2: ", multiple_linear_regression.coef_)

# predict
y_pred = multiple_linear_regression.predict(x_test)

print("r_square score", r2_score(y_test, y_pred))
