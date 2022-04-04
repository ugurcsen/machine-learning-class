import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

x_l = np.load('sign_language_digits_dataset/X.npy')
Y_l = np.load('sign_language_digits_dataset/Y.npy')

img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')
plt.show()

# Prepare labels
y = []
for i in range(Y_l.shape[0]):
    if Y_l[i][0] == 1:
        y.append(0)
    elif Y_l[i][1] == 1:
        y.append(1)
    elif Y_l[i][2] == 1:
        y.append(2)
    elif Y_l[i][3] == 1:
        y.append(3)
    elif Y_l[i][4] == 1:
        y.append(4)
    elif Y_l[i][5] == 1:
        y.append(5)
    elif Y_l[i][6] == 1:
        y.append(6)
    elif Y_l[i][7] == 1:
        y.append(7)
    elif Y_l[i][8] == 1:
        y.append(8)
    elif Y_l[i][9] == 1:
        y.append(9)

x = x_l.reshape(x_l.shape[0], -1)
# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

logisticRegression = LogisticRegression(max_iter=999999)
ovo = OneVsOneClassifier(logisticRegression)
ovr = OneVsRestClassifier(logisticRegression)
ovo.fit(x_train, y_train)
ovr.fit(x_train, y_train)

y_pred_ovo = ovo.predict(x_test)
y_pred_ovr = ovr.predict(x_test)
cf_ovo = confusion_matrix(y_test, y_pred_ovo)
cf_ovr = confusion_matrix(y_test, y_pred_ovr)

print(cf_ovo)
# Output
"""
[[45  0  2  0  1  1  3  1  0  1]
 [ 0 49  0  1  0  1  4  0  0  0]
 [ 0  1 43  0  1  1  3  0  5  1]
 [ 0  0  0 39  1  2  4  0  1  0]
 [ 1  0  1  0 46  3  0  0  1  0]
 [ 3  0  2  3  0 41  2  0  0  0]
 [ 1  0  5  3  0  3 38  0  0  1]
 [ 0  0  1  0  1  1  0 47  2  3]
 [ 0  1  3  2  3  0  0  0 33  0]
 [ 1  0  2  0  0  0  1  3  0 47]]
"""
print("Accuracy OvO:", metrics.accuracy_score(y_test, y_pred_ovo))
# Output
"""
Accuracy OvO: 0.8294573643410853
"""
print("Precision OvO:", metrics.precision_score(y_test, y_pred_ovo, average='micro'))
# Output
"""
Precision OvO: 0.8294573643410853
"""
print("Recall OvO:", metrics.recall_score(y_test, y_pred_ovo, average='micro'))
# Output
"""
Recall OvO: 0.8294573643410853
"""


print(cf_ovr)
# Output
"""
[[42  0  2  1  1  5  1  2  0  0]
 [ 1 45  0  1  0  2  6  0  0  0]
 [ 0  0 39  2  0  2  6  0  5  1]
 [ 0  1  1 33  0  2  5  0  4  1]
 [ 0  0  2  0 41  5  0  1  3  0]
 [ 4  1  1  2  1 36  5  0  1  0]
 [ 1  0  7  5  0  5 29  1  1  2]
 [ 1  0  3  0  1  1  0 43  2  4]
 [ 1  2  5  4  1  1  0  1 27  0]
 [ 2  0  0  0  0  1  0  4  0 47]]
"""
print("Accuracy OvR:", metrics.accuracy_score(y_test, y_pred_ovr))
# Output
"""
Accuracy OvR: 0.7403100775193798
"""
print("Precision OvR:", metrics.precision_score(y_test, y_pred_ovr, average='micro'))
# Output
"""
Precision OvR: 0.7403100775193798
"""
print("Recall OvR:", metrics.recall_score(y_test, y_pred_ovr, average='micro'))
# Output
"""
Recall OvR: 0.7403100775193798
"""
print(ovo.predict(x[[409]]))
# 2
