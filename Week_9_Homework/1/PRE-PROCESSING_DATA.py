import pandas as pd

# Load data
from sklearn.model_selection import train_test_split, GridSearchCV

data = pd.read_csv('./datasets/column_2C_weka.csv')
# get_dummies
df = pd.get_dummies(data)
print(df.head(10).to_string())

# drop one of the feature
# Delete unnecessary colum
df.drop("class_Normal", axis=1, inplace=True)
print(df.head(10).to_string())

# abnormal = 1 and normal = 0
data['class_binary'] = [1 if i == 'Abnormal' else 0 for i in data.loc[:, 'class']]
x, y = data.loc[:, (data.columns != 'class') & (data.columns != 'class_binary')], data.loc[:, 'class_binary']
# instead of two steps we can make it with one step pd.get_dummies(data,drop_first = True)

# SVM, pre-process and pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define preprocessing steps for pipeline
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters = {'SVM__C': [1, 10, 100],
              'SVM__gamma': [0.1, 0.01]}

# Split dataset to two part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
# Find best SVC parameters
cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)
# Fit svc with best tuned parameters
cv.fit(x_train, y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
