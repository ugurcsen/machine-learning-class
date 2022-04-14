import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# importing datasets
from sklearn.svm import SVC

data = pd.read_csv("datasets/data.csv")
print(data.head())
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# Preparing label and futures
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values  # Select labels
x_data = data.drop(["diagnosis"], axis=1)  # Select only futures

# %%
# normalization  ortalama yöntemiyle ayarlıyor
x = (x_data - x_data.min()) / (x_data.max() - x_data.min())


# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# Find best c value for svc model
# find c value
score_list = []
for each in range(1, 100):
    svc = SVC()
    svc.C = each
    svc.fit(x_train, y_train)
    score_list.append(svc.score(x_test, y_test))

print("Best score is {} with C = {}".format(np.max(score_list), 1 + score_list.index(np.max(score_list))))

plt.plot(range(1, 100), score_list)
plt.xlabel("c values")
plt.ylabel("accuracy")
plt.show()

y_pred = svc.predict(x_test)
y_true = y_test

# confusion matrix creating
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

# cm visualization

f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
