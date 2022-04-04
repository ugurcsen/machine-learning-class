# Data Pre-procesing Step
# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# importing datasets
data_set = pd.read_csv('User_Data.csv')

# Extracting Independent and dependent Variable
x = data_set.iloc[:, [2, 3]].values
y = data_set.iloc[:, 4].values

# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)  # transform for train set
x_test = st_x.transform(x_test)  # transform for test set

# Fitting Logistic Regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)  # Training

# Predicting the test set result
y_pred = classifier.predict(x_test)  # Testing

# Creating the Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualizing the training set result
x_set, y_set = x_train, y_train
x1, x2 = nm.meshgrid(nm.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     nm.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('purple', 'green')))
mtp.xlim(x1.min(), x1.max())
mtp.ylim(x2.min(), x2.max())
for i, j in enumerate(nm.unique(y_set)):
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=[ListedColormap(('purple', 'green'))(i)], label=j)  # Error in c fixed with [] brackets
mtp.title('Logistic Regression (Training set)')
mtp.xlabel('Age')
mtp.ylabel('Estimated Salary')
mtp.legend()
mtp.show()
