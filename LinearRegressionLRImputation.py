# Created by: Gerald Jones
# Purpose: This will hopefully use linear regression to model how
#         the number of cylinders, displacement, horsepower, weight,
#         acceleration, model year, and origin of a car can be used with
#         linear regression techniques to predict mpg. This one discards
#         observations with bad data


# import timeit
# import pandas as pd
# from matplotlib.pyplot import *
import numpy as np
import GDataWorks
import DataCleaner
import DataProcessor
import RegressionTools


# stores the name of a column of attributes
# where the index of the attribute matches the column
# it is related to
attribute_label_array = ['mpg',           # 0
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
# car_type_array = GDataWorks.get_car_types(list(dataarray))

# Find the bad data and store it in a map keyed on the column
# where the bad data was found and with the rows in that column
# where the bad data is as the values as a list
# baddatdic = GDW.FindColBadData(dataarray.copy(), '?')
bad_dat_dic = DataProcessor.find_col_bad_data(dataarray.copy(), '?')

# print(bad_dat_dic)


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


# convert string versions of the data sets to into numerical values(either discrete or continuous
# give it the bad/missing signifier and it will find and replace with an integer 0
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


d_a_known = DataProcessor.remove_row(list(dataarray), bad_dat_dic[3])

x_a_lr_i, y_a_lr_i = DataProcessor.x_y_getter(list(d_a_known), attribute_label_array.index('Horse Power'))

print('x_a_lr')
print(x_a_lr_i)
print(len(x_a_lr_i))
print('y_a_lr')
print(y_a_lr_i)
print(len(y_a_lr_i))

min_col, min_mse, best_col = RegressionTools.find_first(list(x_a_lr_i), list(y_a_lr_i), list([.60, .40]))

print('minimum col')
print(min_col[0])
print('minimuc mse')
print(min_mse[0])
print('best_col')
print(best_col[0])


#hp_array = DataProcessor.column_getter(list(y_a_lr_i), attribute_label_array.index('Horse Power'))
hp_array = list(y_a_lr_i)
weight_array = DataProcessor.column_getter(list(x_a_lr_i), min_col[0])

print(format('\n'))
print('Horse Power column')
print(hp_array)
print(len(hp_array))
print('Weight column')
print(weight_array)
print(len(weight_array))
print(format('\n'))


print('length of original data array')
print(dataarray)
print(len(dataarray))

#m, b, x, y, yg, mse = RegressionTools.reg_lin_regression_msr(weight_array, hp_array, [.60, .40])

w_imp = RegressionTools.multi_linear_regressor(x_a_lr_i, hp_array)

rt = DataProcessor.getlinregmissingdata(list(dataarray), bad_dat_dic, w_imp)

print('rt')
print(rt)


imputated_data = DataProcessor.ReplaceBadDatavec(list(dataarray), bad_dat_dic, rt)

for entry in bad_dat_dic:
    for row in bad_dat_dic[entry]:
        print(imputated_data[row][3])
        print(imputated_data[row])


#lr_i_dataarray = DataProcessor.linear_regression_replacer(dataarray, m, b, attribute_label_array.index("Weight"), 3, bad_dat_dic)

# b_d_list = bad_dat_dic[3]


print(format('\n'))
#for i in range(len(b_d_list)):
#print(lr_i_dataarray[b_d_list[i]][3])
#print('length of lr imputation replacement array')
#print(len(lr_i_dataarray))
