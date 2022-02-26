import numpy as np  # import numpy library

# One dimensional array
one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)

# Two dimensional array
two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)

# Create sequenced array
sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)

# Random integer array between 50,100
random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6))
print(random_integers_between_50_and_100)

# Random float array between 0,1
random_floats_between_0_and_1 = np.random.random([6])
print(random_floats_between_0_and_1)

# Random floats array created between 2 and 3
random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print(random_floats_between_2_and_3)

# Random integer array created between 150 and 300
random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3
print(random_integers_between_150_and_300)

# TASK 1: Create Linear dataset
## Assign a sequence of integers from 6 to 20 (inclusive) to a NumPy array named feature.
## Assign 15 values to a NumPy array named label
feature = np.arange(6, 21)  # write your code here
print(feature)
label = (3 * feature + 4)  # write your code here
print(label)

# Task 2: Add Some Noise to the Dataset
# To make your dataset a little more realistic, insert a little random noise into each element of the label array you already created. To be more precise, modify each value assigned to label by adding a different random floating-point value between -2 and +2.
#
# Don't rely on broadcasting. Instead, create a noise array having the same dimension as label.
noise = np.random.random(15) * 4 - 2  # write your code here
print(noise)
label = label + noise  # write your code here
print(label)
