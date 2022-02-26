# create dictionary and look its keys and values
import numpy as np
import pandas as pd

dictionary = {'spain': 'madrid', 'usa': 'vegas', 'turkey': 'istanbul'}
print(dictionary.keys())
print(dictionary.values())

print("=============================================")

# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['spain'] = 'barcelona'  # update existing entry
print(dictionary)
dictionary['france'] = "paris"  # Add new entry
print(dictionary)
del dictionary['spain']  # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)
dictionary[37] = "istanbul"  # add new value with key 37
print(dictionary)
dictionary.clear()  # remove all entries in dict
print(dictionary)

print("=============================================")

# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary
print(dictionary)  # it gives error because dictionary is deleted

data = pd.read_csv('data/pokemon.csv')

series = data['Defense']  # data['Defense'] = series
print(type(series))
data_frame = data[['Defense']]  # data[['Defense']] = data frame show like table
print(type(data_frame))

# Comparison operator
print(3 > 2)
print(3 != 2)
print(3 == '3')
# Boolean operators
print(True and False)
print(True or False)

print("=============================================")

# 1 - Filtering Pandas data frame
x = data['Defense'] > 200  # There are only 3 pokemons who have higher defense value than 200
print(data[x])

# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100
print(data[np.logical_and(data['Defense'] > 200, data['Attack'] > 100)])
print(data[(data['Defense'] > 200) & (data['Attack'] > 100)])  # same thing like np.logical

print("=============================================")

# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5:
    print('i is: ', i)
    i += 1
print(i, ' is equal to 5')

print("=============================================")

# Stay in loop if condition( i is not equal 5) is true
lis = ["a", "b", "c", "d", "e"]
for i in lis:
    print('i is: ', i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):
    print(index, " : ", value)
print('')

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'spain': 'madrid', 'france': 'paris'}
for key, value in dictionary.items():
    print(key, " : ", value)
print('')

# For pandas we can achieve index and value
for index, value in data[['Attack']][0:1].iterrows():
    print(index, " : ", value)
