import csv
import pandas as pd
import seaborn
from matplotlib import pyplot as plt

data = pd.read_csv("car_details_v3.csv")

# Engine
data["engine"] = data["engine"].str.replace(" CC", "")
data["engine"] = data["engine"].astype(float, errors="raise")
# Max power
data["max_power"] = data["max_power"].str.replace(" bhp", "")
data[data["max_power"] == ""] = None
data["max_power"] = data["max_power"].astype(float, errors="raise")
# Mileage
data["mileage"] = data["mileage"].str.replace(" kmpl", "")
data["mileage"] = data["mileage"].str.replace(" km/kg", "")
data[data["mileage"] == ""] = None
data["mileage"] = data["mileage"].astype(float, errors="raise")
# Owner
data["owner"] = data["owner"].str.replace("First Owner", "0")
data["owner"] = data["owner"].str.replace("Second Owner", "1")
data["owner"] = data["owner"].str.replace("Third Owner", "2")
data["owner"] = data["owner"].str.replace("Fourth & Above Owner", "3")
data["owner"] = data["owner"].str.replace("Test Drive Car", "-1")
data["owner"] = data["owner"].astype(float, errors="raise")

print(data.head().to_string())

print(data.info())
f, ax = plt.subplots(figsize=(18, 18))
seaborn.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()

data.plot(kind='scatter', x='max_power', y='selling_price', alpha=0.5, color='red')
plt.show()

data.to_csv("car_details_model_ready.csv")
