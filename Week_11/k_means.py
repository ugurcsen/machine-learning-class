import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# %% create dataset

# class1
x1 = np.random.normal(25, 5, 1000)
y1 = np.random.normal(25, 5, 1000)

# class2
x2 = np.random.normal(55, 5, 1000)
y2 = np.random.normal(60, 5, 1000)

# class3
x3 = np.random.normal(55, 5, 1000)
y3 = np.random.normal(15, 5, 1000)

x = np.concatenate((x1, x2, x3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)

dictionary = {"x": x, "y": y}

data = pd.DataFrame(dictionary)

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x3, y3)
plt.title("Generated clusters")
plt.show()

# K-Means algorithm will see this
plt.scatter(x1, y1, color="black")
plt.scatter(x2, y2, color="black")
plt.scatter(x3, y3, color="black")
plt.title("Generated dataset")
plt.show()

# %% KMEANS

wcss = []

# Finding elbow
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# Plotting graphic to find elbow point
plt.plot(range(1, 15), wcss)
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()
# Elbow point is 3


# %% model for k = [1-6]
for k in range(1, 6):
    kmeans2 = KMeans(n_clusters=k)
    clusters = kmeans2.fit_predict(data)

    data["label"] = clusters

    plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color="red")
    plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color="green")
    plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color="blue")
    plt.scatter(data.x[data.label == 3], data.y[data.label == 3], color="yellow")
    plt.scatter(data.x[data.label == 4], data.y[data.label == 4], color="orange")
    plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], color="black")
    plt.title("Calculated clusters for k = " + str(k))
    plt.show()
