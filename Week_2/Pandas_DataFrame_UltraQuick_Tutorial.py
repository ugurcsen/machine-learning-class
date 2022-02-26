# Import libraries
import numpy as np
import pandas as pd

# Create and populate a 5x2 NumPy array.
my_data = np.array([
    [0, 3],
    [10, 7],
    [20, 9],
    [30, 14],
    [40, 15]
])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)


# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire DataFrame
print(my_dataframe)

print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

print("Column 'temperature':")
print(my_dataframe['temperature'])


# Task 1: Create a DataFrame
# Do the following:
#
# Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns are named Eleanor, Chidi, Tahani,
# and Jason. Populate each of the 12 cells in the DataFrame with a random integer between 0 and 100, inclusive.
#
# Output the following:
#
# the entire DataFrame
# the value in the cell of row #1 of the Eleanor column
# Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason.
#
# To complete this task, it helps to know the NumPy basics covered in the NumPy UltraQuick Tutorial.

npArr = np.random.randint(0, 101, (3, 4))
print(npArr)
cols = ['Eleanor', 'Chidi', 'Tahani', 'Jason']
pDataFrame = pd.DataFrame(data=npArr, columns=cols)
print(pDataFrame)

print('Eleanor Row 1:')
print(pDataFrame['Eleanor'][1])
pDataFrame['Janet'] = pDataFrame['Tahani'] + pDataFrame['Jason']
print(pDataFrame)
