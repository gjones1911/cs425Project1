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

print('dataarray')
print(dataarray)

x_array, y_array = DataProcessor.x_y_getter(list(dataarray), 0)
print("mpg")
print(y_array)
print("Independent Attributes")
print(x_array)


normalized_data = DataProcessor.normalize_data(list(dataarray), smu, std, min_array, max_array, continuous_discrete)

print('normalized data')
print(attribute_label_array)
print(normalized_data)

x_array_n, y_array_n = DataProcessor.x_y_getter(dataarray, 0)

print(format('\n'))
print('normalized data')
print("mpg")
print(y_array_n)
print("Independent Attributes")
print(x_array_n)
print(format('\n'))

print('original data')
print("mpg")
print(y_array)
print("Independent Attributes")
print(x_array)
print(format('\n'))

split_array1 = [[.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40],
                [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40],
                [.60, .40]]

split_array2 = [[.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25],
                [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25],
                [.75, .25]]

split_array3 = [[.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30],
                [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30],
                [.70, .30]]

split_array4 = [[.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50],
                [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50],
                [.50, .50]]

test_l = int(398*.60)
print(test_l)
print(int(398*.40))
print(398 - test_l)

# ----------------------------------------with regulur data------------------------------------------------------------
print(format('\n'))
print('------------------------------------------with regular data--------------------------------------------------')
# Coefficient of Determination :
best_cod, bs_cod, cod_list, rand_list_cod2, best_rand_cod2 = RegressionTools.train_model_cod_dos(list(x_array), list(y_array), split_array1)
print('Coefficient of Determination')
print('best_cod Cod')
print(best_cod)
print('best split Cod')
print(bs_cod)
print("cod list")
print(cod_list)
print('Average error cod')
print(np.around(np.mean(np.array(cod_list, dtype=np.float), dtype=np.float), 2))
# print('Best rand cod 2')
# print(best_rand_cod2)
# print('rand list cod 2')
# print(rand_list_cod2)
print(format("\n"))

# Least Square Error
best_lse, bs_lse, lse_list, rand_list_lse2, best_rand_lse2 = RegressionTools.train_model_lse_dos(list(x_array), list(y_array), split_array1)
print('Least Square Error')
print('best lse ')
print(best_lse)
print('best split lse')
print(bs_lse)
print("lse list")
print(lse_list)
print('Average error lse')
print(np.around(np.mean(np.array(lse_list, dtype=np.float), dtype=np.float), 2))
# print('Best rand lse 2')
# print(best_rand_lse2)
# print('rand list lse 2')
# print(rand_list_lse2)
print(format("\n"))

# Mean Square Error :
best_mse, bs_mse, mse_list, rand_list_mse2, best_rand_mse2 = RegressionTools.train_model_mse_dos(list(x_array), list(y_array), split_array1)
print('mean Square Error')
print('best mse ')
print(best_mse)
print('best split mse')
print(bs_mse)
print("mse list")
print(mse_list)
print('Average error mse')
print(np.around(np.mean(np.array(mse_list, dtype=np.float), dtype=np.float), 2))
# print('Best rand lse 2')
# print(best_rand_lse2)
# print('rand list lse 2')
# print(rand_list_lse2)
print('---------------------------------------end with regular data--------------------------------------------------')
print(format("\n"))

# ----------------------------------------with normalized data---------------------------------------------------------
print(format('\n'))
print('------------------------------------------with normalized data------------------------------------------------')
# cod
best_cod_n, bs_cod_n, cod_list_n, rand_list_cod2_n, best_rand_cod2_n = RegressionTools.train_model_cod_dos(list(x_array_n), list(y_array_n), split_array1)
print('Coefficient of Determination normalized')
print('best_cod Cod normalized')
print(best_cod_n)
print('best split Cod normalized')
print(bs_cod_n)
print("cod list normalized")
print(cod_list_n)
print('Average error cod normalized')
print(np.around(np.mean(np.array(cod_list_n, dtype=np.float), dtype=np.float), 2))
# print('Best rand cod 2')
# print(best_rand_cod2)
# print('rand list cod 2')
# print(rand_list_cod2)
print(format("\n"))

# Least Square Error :
best_lse_n, bs_lse_n, lse_list_n, rand_list_lse2_n, best_rand_lse2_n = RegressionTools.train_model_lse_dos(list(x_array_n), list(y_array_n), split_array1)
print('Least Square Error norm')
print('best lse norm')
print(best_lse_n)
print('best split lse norm')
print(bs_lse_n)
print("lse list norm")
print(lse_list_n)
print('Average error lse norm')
print(np.around(np.mean(np.array(lse_list_n, dtype=np.float), dtype=np.float), 2))
# print('Best rand lse 2')
# print(best_rand_lse2)
# print('rand list lse 2')
# print(rand_list_lse2)
print(format("\n"))

# Mean Square Error :
best_mse_n, bs_mse_n, mse_list_n, rand_list_mse2_n, best_rand_mse2_n = RegressionTools.train_model_mse_dos(list(x_array_n), list(y_array_n), split_array1)
print('mean Square Error norm')
print('best mse norm')
print(best_mse_n)
print('best split mse norm')
print(bs_mse_n)
print("mse list norm")
print(mse_list_n)
print('Average error mse norm')
print(np.around(np.mean(np.array(mse_list_n, dtype=np.float), dtype=np.float), 2))
# print('Best rand lse 2')
# print(best_rand_lse2)
# print('rand list lse 2')
# print(rand_list_lse2)
print(format("\n"))



print('---------------------------------------end with regular data--------------------------------------------------')
print(format("\n"))





F, best_mse = RegressionTools.forward_selector(list(x_array), list(y_array), split_array1[0])

print('best MSR is ')
print(best_mse)
print('F')
print(len(F))
print(len(F[0]))
print(F)
print(format('\n'))


print('normalized data')
F_n, best_mse_n = RegressionTools.forward_selector(list(x_array_n), list(y_array_n), split_array1[0])

print('best MSR norm is ')
print(best_mse_n)
print('F norm')
print(len(F_n))
print(len(F_n[0]))
print(F_n)
print(format('\n'))


# test = [1,2,3,4,5]
# for i in range(0, 10):
#    random_selection = np.random.choice(test, 5, replace=False)
#    print(random_selection)
