'''
Created by: Gerald Jones
Purpose: This will hopefully use linear regression to model how
         the number of cylinders, displacement, horsepower, weight,
         acceleration, model year, and origin of a car can be used with
         linear regression techniques to predict mpg. This one discards
         observations with bad data
'''

import timeit
import pandas as pd
import numpy as np
from matplotlib.pyplot import *
import GDataWorks as GDW
import DataCleaner as DC

error_array = ['COD', 'Least_squares', 'MSE']

# stores the name of a column of attributes
# where the index of the attribute matches the column
# it is related to
attribute_type_array = [''
                        'mpg',           # 0
                        'Cylinders',     # 1
                        'Displacement',  # 2
                        'Horse Power',   # 3
                        'Weight',        # 4
                        'Acceleration',  # 5
                        'Model Year',    # 6
                        'Origin',        # 7
                        'Car Type']      # 8

attribute_label_array = ['mpg',          # 0
                        'Cylinders',     # 1
                        'Displacement',  # 2
                        'Horse Power',   # 3
                        'Weight',        # 4
                        'Acceleration',  # 5
                        'Model Year',    # 6
                        'Origin',        # 7
                        'Car Type']      # 8

continous_discrete = [0,   # 0 mpg
                      1,   # 1 cylinders
                      0,   # 2 displacement
                      0,   # 3 horse power
                      0,   # 4 weight
                      0,   # 5 acceleration
                      1,   # 6 model year
                      1,   # 7 Origin
                      1,]  # 8 car type number

# get the data using data cleaner
# returns a 2D array where rows are observations and columns
# are attributes of a specific observations
dataarray = DC.DataCleaner("CarData.txt")


# returns a list of different car types
car_type_array = GDW.get_car_types(list(dataarray))

# Find the bad data and store it in a map keyed on the column
# where the bad data was found and with the rows in that column
# where the bad data is as the values as a list
baddatdic = GDW.FindColBadData(dataarray.copy(), '?')

# returns a dictionary where the keys are car names and the values
# are the number that represents that car type/name
# also returns a list of the numbers that represent the different car names/types
name_dic, name_to_num = GDW.convert_nominal_to_int(car_type_array)


# replace the names with reference numbers
e_end = len(dataarray[0]) - 1
for idx in range(len(dataarray)):
    dataarray[idx][e_end] = name_to_num[idx]

'''
print(format("\n"))
print(dataarray)
print(len(dataarray[0]))
print(format("\n"))
'''
count = 0


#convert string versions of the data sets to into numerical values(either discrete or continous)
for row in range(len(dataarray)-count):
    #print(row)
    for col in range(len(dataarray[0])):
        val = dataarray[row][col]
        if val == '?':
            dataarray[row][col] = float(0)
            #del dataarray[row]
            #count += 1
        else:
            if continous_discrete[col] == 1:
                dataarray[row][col] = int(dataarray[row][col])
            else:
                dataarray[row][col] = float(dataarray[row][col])

#print(dataarray)
#print('dataarray')
#print(dataarray)
#print(format("\n"))
#print('dataarray length')
#print(len(dataarray))
#print(format("\n"))
#print(baddatdic[3])

# remove rows with bad/missing data
for entry in baddatdic:
    bad_list = baddatdic[entry]
    for row in bad_list:
        #print('deleting row: ', row)
        del dataarray[row]
        del car_type_array[row]
'''
print('dataarray')
print(dataarray)
print(format("\n"))
print('dataarray length')
print(len(dataarray))
print(format("\n"))
'''

np_dataarray = np.array(dataarray, dtype=np.float)

dataarray_df = pd.DataFrame(data=dataarray,
                            index=car_type_array,
                            columns=attribute_type_array)

# print(format("\n"))
# print(dataarray_df)
# print(format("\n"))
# print(dataarray_df.loc[:,'Model Year'])
#print(format("\n"))
#print(GDW.column_getter(dataarray, attribute_label_array.index('mpg')))

# calculate mean array

smu = GDW.sample_mean_array(dataarray, continous_discrete)

print('sample mean array')
print(smu)

std = GDW.sample_std_array2(dataarray)
print('sample std array')
print(std)



tester = [[1, 2, 3, 4, 9],
          [5, 6, 7, 8, 10]]

#hlf1 = [tester[i][0:2] for i in range(0, len(tester))]
#endr = [tester[i][2+1:] for i in range(0, len(tester))]

#chunk1 = hlf1[0] + endr[0]
#chunk2 = hlf1[1] + endr[1]
#chunk = [chunk1] + [chunk2]
#print(chunk)

ycolumn = int(2)

xx, yy = GDW.x_y_getter(dataarray, attribute_label_array.index('mpg'))
print('X array')
print(xx)

print('Y array')
print(yy)
print(format('\n'))


best_cod, bs = GDW.TrainModelCOD(xx, yy)
print('best_cod Cod')
print(best_cod)
print('best split Cod')
print(bs)
#t = timeit.Timer("x_y_getter(tester, ycolumn)", "from  GDataWorks import x_y_getter, tester, ycolumn")
#t = timeit.Timer("x_y_getter(tester, ycolumn)", "from  GDataWorks import x_y_getter",  "from LinearRegressionDiscard import tester, ycolumn")
#print(t.timeit(5))