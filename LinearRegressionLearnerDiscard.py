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
# from matplotlib.pyplot import *
import GDataWorks as GDW
import DataCleaner as DC
import DataProcessor
import RegressionTools

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
dataarray = DC.DataCleaner("CarData.txt")


# returns a list of different car types
# car_type_array = GDW.get_car_types(list(dataarray))

# Find the bad data and store it in a map keyed on the column
# where the bad data was found and with the rows in that column
# where the bad data is as the values as a list
baddatdic = GDW.FindColBadData(dataarray.copy(), '?')

# remove the column for car name
dataarray = DataProcessor.remove_col(list(dataarray), 8)

# Convert strings to numerical values
# using the continuous/discrete array to turn the value into a float or an int respectively
dataarray = DataProcessor.convert_strings_float_int(list(dataarray), '?',  continuous_discrete)

# remove the rows with bad data
dataarray = DataProcessor.remove_row(list(dataarray), baddatdic[3])

# returns a dictionary where the keys are car names and the values
# are the number that represents that car type/name
# also returns a list of the numbers that represent the different car names/types
# name_dic, name_to_num = GDW.convert_nominal_to_int(car_type_array)


# replace the names with reference numbers
# e_end = len(dataarray[0]) - 1
# for idx in range(len(dataarray)):
#     dataarray[idx][e_end] = name_to_num[idx]

'''
print(format("\n"))
print(dataarray)
print(len(dataarray[0]))
print(format("\n"))
'''

#count = 0
#convert string versions of the data sets to into numerical values(either discrete or continous)
#for row in range(len(dataarray)-count):
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

#print(dataarray)
#print('dataarray')
#print(dataarray)
#print(format("\n"))
#print('dataarray length')
#print(len(dataarray))
#print(format("\n"))
#print(baddatdic[3])

# remove rows with bad/missing data

#for entry in baddatdic:
#    bad_list = baddatdic[entry]
#    for row in bad_list:
#        #print('deleting row: ', row)
#        del dataarray[row]
#        del car_type_array[row]

'''
print('dataarray')
print(dataarray)
print(format("\n"))
print('dataarray length')
print(len(dataarray))
print(format("\n"))
'''

#np_dataarray = np.array(dataarray, dtype=np.float)

#dataarray_df = pd.DataFrame(data=dataarray,
#                            index=car_type_array,
#                            columns=attribute_type_array)

# print(format("\n"))
# print(dataarray_df)
# print(format("\n"))
# print(dataarray_df.loc[:,'Model Year'])
#print(format("\n"))
#print(GDW.column_getter(dataarray, attribute_label_array.index('mpg')))

# calculate mean array

smu = DataProcessor.sample_mean_array(dataarray, continuous_discrete)
print(format('\n'))
print('sample mean array')
print(smu)
print(format('\n'))

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

#tester = [[1, 2, 3, 4, 9],
#          [5, 6, 7, 8, 10]]

#hlf1 = [tester[i][0:2] for i in range(0, len(tester))]
#endr = [tester[i][2+1:] for i in range(0, len(tester))]

#chunk1 = hlf1[0] + endr[0]
#chunk2 = hlf1[1] + endr[1]
#chunk = [chunk1] + [chunk2]
#print(chunk)

x_array, y_array = DataProcessor.x_y_getter(list(dataarray), attribute_label_array.index('mpg'))
print("mpg")
print(y_array)
print("Independent Attributes")
print(x_array)


'''
best_cod, bs = GDW.TrainModelCOD(xx, yy)
print('best_cod Cod')
print(best_cod)
print('best split Cod')
print(bs)
#t = timeit.Timer("x_y_getter(tester, ycolumn)", "from  GDataWorks import x_y_getter, tester, ycolumn")
#t = timeit.Timer("x_y_getter(tester, ycolumn)", "from  GDataWorks import x_y_getter",  "from LinearRegressionDiscard import tester, ycolumn")
#print(t.timeit(5))
'''

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
print(len(y_array))
print("Independent Attributes")
print(x_array)
print(len(x_array))
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

# Least Square Error :
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

# Least Square Error :
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
print(format("\n"))



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