import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

data = pd.read_csv("datasets/data.csv")
print(data.head())
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

# %% tablonun hazırlanışı
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# scatter plot
plt.scatter(M.radius_mean, M.texture_mean, color="red", label="kotu", alpha=0.3)
plt.scatter(B.radius_mean, B.texture_mean, color="green", label="iyi", alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# Preparing label and futures
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values  # Select labels
x_data = data.drop(["diagnosis"], axis=1)  # Select only futures

# %%
# normalization adjust with centralization method
x = (x_data - x_data.min()) / (x_data.max() - x_data.min())

# %%
# train test split train ve test verisini ayırdık
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# %% SVM
svm = SVC(random_state=1)  # Create Model
svm.fit(x_train, y_train)  # Train model

# %% test
print("print accuracy of svm algo: ", svm.score(x_test, y_test))

y_pred = svm.predict(x_test)  # Test
print(y_pred)

# Making the Confusion Matrix
print(confusion_matrix(y_test, y_pred))
