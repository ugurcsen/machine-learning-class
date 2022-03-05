from sklearn import datasets

import matplotlib.pyplot as plt

# Load the digits dataset
digits = datasets.load_digits()
# Display digits
for i in range(10):
    plt.figure(1, figsize=(3, 3))
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()