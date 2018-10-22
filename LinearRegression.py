# Created by: Gerald Jones
# Purpose: This will hopefully use linear regression to model how
#         the number of cylinders, displacement, horsepower, weight,
#         acceleration, model year, and origin of a car can be used with
#         linear regression techniques to predict mpg. This one discards
#         observations with bad data

# import numpy as np
# from matplotlib.pyplot import *
# import GDataWorks
import DataCleaner
import DataProcessor
import RegressionTools

options = ['0', '1', '2',]
Imputations_options = ['0', '1', '2', '3', '4', '5']
imputation_methods = ['Discard Imputation', 'Average Imputation', 'Linear Regression Imputation',
                      'discard with forward selection',
                      'Average Imputation with forward selection',
                      'Linear Regression Imputation with forward selection']
error_methods = ['Coefficient of Determination', 'Least Squares Estimate', 'Mean Square Error']
while True:
    print(format('Imputation options: '))
    print(format("0: Use Discard Imputation               3: Discard Imputation with forward selection"))
    print(format("1: Use Average Imputation               4: Average Imputation with forward selection"))
    print(format("2: Use Linear regression Imputation     5: Linear Regression Imputation with forward selection"))
    imputation = input("Chose a Imputation Method: ")
    if imputation in Imputations_options:
        break
    else:
        print("No option for " + imputation)
        print(format('\n'))


print(str.format('Using ' + imputation_methods[int(imputation)] + '\n'))


while True:
    print(str.format('Error Checking options: \n'))
    print(str.format("0: Use " + error_methods[0]))
    print(str.format("1: Use " + error_methods[1]))
    print(str.format("2: Use " + error_methods[2]))
    error = input("Chose a Error Checking Method: ")
    if error in options:
        break
    else:
        print("No option for " + error)
        print(format('\n'))


print(str.format('Using ' + error_methods[int(error)] + '\n'))

# stores the name of a column of attributes
# where the index of the attribute matches the column
# it is related to
# attribute_type_array = [''
#                        'mpg',  # 0
#                        'Cylinders',  # 1
#                        'Displacement',  # 2
#                        'Horse Power',  # 3
#                        'Weight',  # 4
#                        'Acceleration',  # 5
#                        'Model Year',  # 6
#                        'Origin',  # 7
#                        'Car Type']  # 8

attribute_label_array = ['mpg',  # 0
                         'Cylinders',  # 1
                         'Displacement',  # 2
                         'Horse Power',  # 3
                         'Weight',  # 4
                         'Acceleration',  # 5
                         'Model Year',  # 6
                         'Origin',  # 7
                         'Car Type']  # 8

cont_dis = [0,    # 0 mpg
            1,    # 1 cylinders
            0,    # 2 displacement
            0,    # 3 horse power
            0,    # 4 weight
            0,    # 5 acceleration
            1,    # 6 model year
            1,    # 7 Origin
            1, ]  # 8 car type number

cols_rmv = [8]

# the below arrays are used to run 20 runs with the sam separation(training, validation)
# the different arrays attempt to see if there is an optimal seperation
'''
split_array1 = [[.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40],
                [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40], [.60, .40],
                [.60, .40]]

split_array2 = [[.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25],
                [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25], [.75, .25],
                [.75, .25]]

split_array3 = [[.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30],
                [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30],
                [.70, .30], [.70, .30], [.70, .30], [.70, .30], [.70, .30]]

split_array4 = [[.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50],
                [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50], [.50, .50],
                [.50, .50]]
'''

size1 = [.6 , .4]
size2 = [.75, .25]
size3 = [.7, .3]
size4 = [.5, .5]

split_array1 = list()
split_array2 = list()
split_array3 = list()
split_array4 = list()
limit = 25
for x in range(0,limit):
    split_array1.append(size1)
for x in range(0,limit):
    split_array2.append(size2)

for x in range(0,limit):
    split_array3.append(size3)

for x in range(0,limit):
    split_array4.append(size4)

split_selection = split_array1
#split_selection = split_array2
#split_selection = split_array3
#split_selection = split_array4

# get the data using data cleaner
# returns a 2D array where rows are observations and columns
# are attributes of a specific observations
dataarray = DataCleaner.DataCleaner("CarData.txt")

# print('Data Array')
# print(dataarray)

if int(imputation) == 0:
    print('-----------------------------------')
    print('Using Discard Imputation')
    print('-----------------------------------')
    dataarray, stat_a, x, y, x_n, y_n = DataProcessor.discard_imputation(list(dataarray), cont_dis, cols_rmv, '?', 0)

    # RegressionTools.show_data_x_y(dataarray, x, y)
    # RegressionTools.show_stat_array(stat_a)

    imp = imputation_methods[int(imputation)]
    err = error_methods[int(error)]

    if int(error) == 0 or int(error) == -1 :
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_cod2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_cod2(param_tr_val_a_n)

        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data')
        print(format('\n'))
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)

    if int(error) == 1 or int(error) == -1 :
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_lse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_lse2(param_tr_val_a_n)

        print(format('\n'))
        print('Raw Data')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
    if int(error) == 2 or int(error) == -1 :
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_mse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_mse2(param_tr_val_a_n)

        print('Raw Data')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data')
        print(format('\n'))
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
if int(imputation) == 1 or int(imputation) == -1:
    print('--------------------------------------')
    print('Using Average Imputation')
    print('--------------------------------------')
    print(format('\n'))
    dataarray, stat_a, x, y, x_n, y_n = DataProcessor.average_imputation(list(dataarray), cont_dis, cols_rmv, '?', 0)

    # RegressionTools.show_data_x_y(dataarray, x, y)
    # RegressionTools.show_stat_array(stat_a)

    imp = imputation_methods[int(imputation)]
    err = error_methods[int(error)]

    if int(error) == 0 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_cod2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_cod2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data:')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
    if int(error) == 1 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_lse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_lse2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data:')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
    if int(error) == 2 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_mse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_mse2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
if int(imputation) == 2:
    print('-----------------------------------------')
    print('Using Linear Regression imputation')
    print('-----------------------------------------')
    print(format('\n'))

    dataarray, stat_a, x, y, x_n, y_n = DataProcessor.linear_regression_imputation(list(dataarray), cont_dis, cols_rmv, '?', 0)

    # RegressionTools.show_data_x_y(dataarray, x, y)
    # RegressionTools.show_stat_array(stat_a)

    imp = imputation_methods[int(imputation)]
    err = error_methods[int(error)]

    if int(error) == 0 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_cod2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_cod2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data:')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
    if int(error) == 1 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_lse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_lse2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data:')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
    if int(error) == 2 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_mse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_mse2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data:')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
if int(imputation) == 3:
    print('-----------------------------------')
    print('Using Discard Imputation with forward selection')
    print('-----------------------------------')
    dataarray, stat_a, x, y, x_n, y_n = DataProcessor.discard_imputation(list(dataarray), cont_dis, cols_rmv, '?', 0)

    F, min_mse, cols_f = RegressionTools.forward_selector_test(list(x), list(y), split_selection[0])
    f_n, min_mse_n, cols_f_n = RegressionTools.forward_selector_test(list(x_n), list(y_n), split_selection[0])

    # print('F is now:')
    # print(F)
    # print('cols are')
    # print(cols_f)
    # print('F n is now:')
    # print(f_n)
    # print('cols are')
    # print(cols_f_n)
    print('F is using ' + str(len(F[0]) - 1) + ' attributes')

    # RegressionTools.show_data_x_y(dataarray, x, y)
    # RegressionTools.show_stat_array(stat_a)

    imp = imputation_methods[int(imputation)]
    err = error_methods[int(error)]

    if int(error) == 0 or int(error) == -1 :
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(F, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_cod2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(f_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_cod2(param_tr_val_a_n)

        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data')
        print(format('\n'))
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)

    if int(error) == 1 or int(error) == -1 :
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(F, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_lse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(f_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_lse2(param_tr_val_a_n)

        print(format('\n'))
        print('Raw Data')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
    if int(error) == 2 or int(error) == -1 :
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(F, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_mse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(f_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_mse2(param_tr_val_a_n)

        print('Raw Data')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data')
        print(format('\n'))
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
if int(imputation) == 4:
    print('--------------------------------------')
    print('Using Average Imputation with forward selection:')
    print('--------------------------------------')
    print(format('\n'))
    dataarray, stat_a, x, y, x_n, y_n = DataProcessor.average_imputation(list(dataarray), cont_dis, cols_rmv, '?', 0)

    F, min_mse, cols_f = RegressionTools.forward_selector_test(list(x), list(y), split_selection[0])
    f_n, min_mse_n, cols_f_n = RegressionTools.forward_selector_test(list(x_n), list(y_n), split_selection[0])

    print('F is using ' + str(len(F[0]) - 1) + ' attributes')

    # RegressionTools.show_data_x_y(dataarray, x, y)
    # RegressionTools.show_stat_array(stat_a)

    imp = imputation_methods[int(imputation)]
    err = error_methods[int(error)]

    if int(error) == 0 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(F, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_cod2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(f_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_cod2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data:')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
    if int(error) == 1 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_lse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_lse2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data:')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
    if int(error) == 2 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_mse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_mse2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
if int(imputation) == 5:
    print('-----------------------------------------')
    print('Using Linear Regression imputation with forward Selection')
    print('-----------------------------------------')
    print(format('\n'))

    dataarray, stat_a, x, y, x_n, y_n = DataProcessor.linear_regression_imputation(list(dataarray), cont_dis, cols_rmv, '?', 0)

    F, min_mse, cols_f = RegressionTools.forward_selector_test(list(x), list(y), split_selection[0])
    f_n, min_mse_n, cols_f_n = RegressionTools.forward_selector_test(list(x_n), list(y_n), split_selection[0])
    print('F is using ' + str(len(F[0]) - 1) + ' attributes')
    # RegressionTools.show_data_x_y(dataarray, x, y)
    # RegressionTools.show_stat_array(stat_a)

    imp = imputation_methods[int(imputation)]
    err = error_methods[int(error)]

    if int(error) == 0 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(F, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_cod2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(f_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_cod2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data:')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
    if int(error) == 1 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_lse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_lse2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data:')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)
    if int(error) == 2 or int(error) == -1:
        w_list, tr_l, y_tr_l, val_l, y_val_l = RegressionTools.collect_parameters2(x, y, split_selection)
        param_tr_val_a = [w_list, tr_l, y_tr_l, val_l, y_val_l]
        ret_list = RegressionTools.train_model_mse2(param_tr_val_a)

        w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n = RegressionTools.collect_parameters2(x_n, y_n, split_selection)
        param_tr_val_a_n = [w_list_n, tr_l_n, y_tr_l_n, val_l_n, y_val_l_n]
        ret_list2 = RegressionTools.train_model_mse2(param_tr_val_a_n)

        print('Raw Data:')
        RegressionTools.show_test_results(imp, err, ret_list, 4, 2)
        print(format('\n'))
        print('Normalized Data:')
        RegressionTools.show_test_results(imp, err, ret_list2, 4, 2)