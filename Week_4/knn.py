# KNN Algorithm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV data file read
data = pd.read_csv("data.csv")

# Drop columns which id and unnamed
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
# data.tail()
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

# Data split to two different part
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# scatter plot
plt.scatter(M.radius_mean, M.texture_mean, color="red", label="Bad", alpha=0.4)
plt.scatter(B.radius_mean, B.texture_mean, color="green", label="Good", alpha=0.4)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.title("Confusion Matrix")
plt.legend()
plt.show()

# Change M to 1 and B to 0
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# Feature scaling
# normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# Train and test set creating
# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# KNN model creating with k = 3
# knn model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)  # n_neighbors = k
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(3, knn.score(x_test, y_test)))

# Find best k value for knn model
# find k value
score_list = []
for each in range(1, 15):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test, y_test))

plt.plot(range(1, 15), score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

y_pred = knn2.predict(x_test)
y_true = y_test

# confusion matrix creating
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

# cm visualization
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
