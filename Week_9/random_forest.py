import pandas as pd

# %%  import data

data = pd.read_csv("data.csv")
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# Set M to 1 and other to 0
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)
# %% normalization

x = (x_data - x_data.min()) / (x_data.max() - x_data.min())

# %% train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# %% decision tree
from sklearn.tree import DecisionTreeClassifier

# Definition of decision tree
dt = DecisionTreeClassifier()
# Train decision tree
dt.fit(x_train, y_train)
# Test
print("decision tree score: ", dt.score(x_test, y_test))

# %%  random forest
from sklearn.ensemble import RandomForestClassifier

# Find the best estimator count
# Best result is 4
for i in range(1, 100):
    rf = RandomForestClassifier(n_estimators=i, random_state=1)
    rf.fit(x_train, y_train)
    print("random forest algo result(i={}): ".format(i), rf.score(x_test, y_test))
