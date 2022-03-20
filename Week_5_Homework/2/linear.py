import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("car_details_model_ready.csv")
data = data.drop(
    ['Unnamed: 0', 'name', 'year', 'seller_type', 'km_driven', 'fuel', 'transmission', 'owner', 'mileage', 'engine',
     'torque', 'seats'], axis=1)
imputer = SimpleImputer(strategy="median")  #
imputer.fit(data)

data2 = imputer.transform(data)  # Fill missing values

data = pd.DataFrame(data2, columns=["selling_price", "max_power"])
x = data["max_power"].values.reshape(-1, 1)
y = data["selling_price"].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lr = LinearRegression()

lr.fit(x_train, y_train)

pred = lr.predict(x_test)

data.plot(kind='scatter', x='max_power', y='selling_price', alpha=0.5, color='blue')
plt.plot(x_test, pred, color="red")
plt.show()

print(r2_score(y_test, pred))
