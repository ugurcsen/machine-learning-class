import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# importing datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data = pd.read_csv("datasets/data.csv")
print(data.head())
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# Preparing label and futures
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values  # Select labels
x_data = data.drop(["diagnosis"], axis=1)  # Select only futures

# %%
# normalization adjust with centralization method
x = (x_data - x_data.min()) / (x_data.max() - x_data.min())

# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

svc = SVC(C=3)
svc.fit(x_train, y_train)
svcScore = svc.score(x_test, y_test)
print("SVC:", svcScore)

gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnbScore = gnb.score(x_test, y_test)
print("GaussianNB:", gnbScore)
