# Created by: Gerald Jones
# Purpose: This will hopefully use linear regression to model how
#         the number of cylinders, displacement, horsepower, weight,
#         acceleration, model year, and origin of a car can be used with
#         linear regression techniques to predict mpg. This one discards
#         observations with bad data


# import timeit
# import pandas as pd
# import numpy as np
from matplotlib.pyplot import *
import GDataWorks
import DataCleaner
import DataProcessor


# stores the name of a column of attributes
# where the index of the attribute matches the column
# it is related to
attribute_label_array = ['mpg',          # 0
                         'Cylinders',     # 1
                         'Displacement',  # 2
                         'Horse Power',   # 3
                         'Weight',        # 4
                         'Acceleration',  # 5
                         'Model Year',    # 6
                         'Origin',        # 7
                         'Car Type']      # 8

# used for calculations to tell if the value
# needed is continous of discrete
continuous_discrete = [0,   # 0 mpg
                      1,   # 1 cylinders
                      0,   # 2 displacement
                      0,   # 3 horse power
                      0,   # 4 weight
                      0,   # 5 acceleration
                      1,   # 6 model year
                      1,   # 7 Origin
                      1, ]  # 8 car type number

# get the data using data cleaner
# returns a 2D array where rows are observations and columns
# are attributes of a specific observations
dataarray = DataCleaner.DataCleaner("CarData.txt")


# returns a list of different car types
car_type_array = GDataWorks.get_car_types(list(dataarray))

# Find the bad data and store it in a map keyed on the column
# where the bad data was found and with the rows in that column
# where the bad data is as the values as a list
# baddatdic = GDW.FindColBadData(dataarray.copy(), '?')
bad_dat_dic = DataProcessor.find_col_bad_data(dataarray.copy(), '?')

print(bad_dat_dic)


# returns a dictionary where the keys are car names and the values
# are the number that represents that car type/name
# also returns a list of the numbers that represent the different car names/types
# name_dic, name_to_num = GDW.convert_nominal_to_int(car_type_array)


# replace the names with reference numbers


#e_end = len(dataarray[0]) - 1
#for idx in range(len(dataarray)):
#    dataarray[idx][e_end] = name_to_num[idx]

'''
print(format("\n"))
print(dataarray)
print(len(dataarray[0]))
print(format("\n"))
'''
# count = 0

# remove the carnames from the array
dataarray = DataProcessor.remove_col(list(dataarray), 8)

#for row in range(0, len(dataarray)):
#    del dataarray[row][8]


print('data array')
print(dataarray)


# convert string versions of the data sets to into numerical values(either discrete or continous)
dataarray = DataProcessor.convert_strings_float_int(list(dataarray), '?',  continuous_discrete)

# for row in range(len(dataarray)):
#    #print(row)
#    for col in range(len(dataarray[0])):
#        val = dataarray[row][col]
#        if val == '?':
#            dataarray[row][col] = float(0)
#            #del dataarray[row]
#            #count += 1
#        else:
#            if continous_discrete[col] == 1:
#                dataarray[row][col] = int(dataarray[row][col])
#            else:
#                dataarray[row][col] = float(dataarray[row][col])


# calculate the average of the HorsePowerData
# Get the horse power column
#
HP_array = DataProcessor.column_getter(dataarray, attribute_label_array.index('Horse Power'))

# make a copy of it so we can average it up
HP_averager = list(HP_array)

# remove the rows with bad data points
HP_averager = DataProcessor.remove_row(HP_averager, bad_dat_dic[3])


# calculate the average of the Horse Power attribute
# and use it to replace missing data
np_HP_averager = np.array(HP_averager, dtype=np.float)
HP_mean = np.mean(np_HP_averager, dtype=np.float)

print(format('\n'))
print('HP\'s average')
print(HP_mean)
print(format('\n'))

#print(dataarray)
#print('dataarray')
#print(dataarray)
#print(format("\n"))
#print('dataarray length')
#print(len(dataarray))
#print(format("\n"))
#print(baddatdic[3])

# remove rows with bad/missing data




'''
for entry in baddatdic:
    bad_list = baddatdic[entry]
    for row in bad_list:
        #print('deleting row: ', row)
        del dataarray[row]
        del car_type_array[row]
'''

value_array = [HP_mean]

dataarray = DataProcessor.replace_item(list(dataarray),bad_dat_dic, value_array)

# replace missing data with the average of the known data
# for entry in bad_dat_dic[3]:
#    dataarray[entry][3] = HP_mean


print('data array using average imputation')
print(dataarray[32])
print(dataarray[126])
print(dataarray[330])
print(dataarray[336])
print(dataarray[354])
print(dataarray[374])


'''
print('dataarray')
print(dataarray)
print(format("\n"))
print('dataarray length')
print(len(dataarray))
print(format("\n"))
'''

'''
np_dataarray = np.array(dataarray, dtype=np.float)

dataarray_df = pd.DataFrame(data=dataarray,
                            index=car_type_array,
                            columns=attribute_type_array)
'''

# print(format("\n"))
# print(dataarray_df)
# print(format("\n"))
# print(dataarray_df.loc[:,'Model Year'])
# print(format("\n"))
# print(GDW.column_getter(dataarray, attribute_label_array.index('mpg')))

# calculate mean array

#calculate the sample mean
#smu = GDataWorks.sample_mean_array(dataarray, continuous_discrete)
smu = DataProcessor.sample_mean_array(dataarray, continuous_discrete)
print(format('\n'))
print('sample mean array')
print(smu)
print(format('\n'))

# std = GDataWorks.sample_std_array2(dataarray)
std = DataProcessor.sample_std_array(dataarray)
print(format('\n'))
print('sample std array')
print(std)
print(format('\n'))

min_array, max_array = DataProcessor.max_min_array_getter(dataarray)
print(format('\n'))
print('Min array')
print(min_array)
print('Max array')
print(max_array)
print(format('\n'))

z_normalized_data = DataProcessor.z_normalization(dataarray, smu, std)

print('Z normalized data')
print(z_normalized_data)
