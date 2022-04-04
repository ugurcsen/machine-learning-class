# Gender Classification
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

cols = ["Favorite_Color", "Favorite_Music_Genre", "Favorite_Beverage", "Favorite_Soft_Drink", "Gender"]
data = pd.read_csv("Transformed_Data_Set.csv", skiprows=1, names=cols)

# Transform string categories to numbers
ordinal_encoder = OrdinalEncoder()
data_encoded = pd.DataFrame(ordinal_encoder.fit_transform(data), columns=cols)

# Features
feature_cols = cols[:-1]
X = data_encoded[feature_cols]
y = data_encoded["Gender"]
# split dataset to two part 75:25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, y_train)

y_pred = logisticRegression.predict(X_test)

# Create confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
# Create heatmap
fig, ax = plt.subplots()
# create heatmap
seaborn.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

print(logisticRegression.score(X_test, y_test))

# Visualizing the training set result
# Print Scores
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

y_pred_proba = logisticRegression.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()
