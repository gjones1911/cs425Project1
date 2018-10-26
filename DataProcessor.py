import numpy as np
import GDataWorks as GDW
# import pandas as pd
# import operator
# from DataCleaner import *
import operator
# local global variables
import RegressionTools
from numpy.core.multiarray import ndarray

a_data = []


# -------------------------------------data splitters------------------------------------------------
# will split the x and y data into training, validation, and test versions
# uses percentages found in split_array traing_percent, validation_percent, test_percent]
def tres_splitter(x_data, y_data, split_array):
    training = []
    y_training = []
    validation = []
    y_validation = []
    test_array = []
    y_test = []

    # used to pick random observations for each data set
    random_selection = np.random.choice(len(x_data), len(y_data), replace=False)

    training_limit = split_array[0]*len(x_data)
    validation_limit = split_array[1]*len(x_data) + training_limit
    test_limit = len(x_data) - validation_limit
    test_limit += validation_limit

    for idx in (0, training_limit):
        row = random_selection[idx]
        # print('adding row: ' + str(row))
        # print(row)
        # #print(format('\n'))
        training.append(x_data[row])
        y_training.append(y_data[row])

    for idx in (training_limit, validation_limit):
        row = random_selection[idx]
        # print('adding row: ' + str(row))
        # print(row)
        # print(format('\n'))
        validation.append(x_data[row])
        y_validation.append(y_data[row])

    for idx in (validation_limit, test_limit):
        row = random_selection[idx]
        # print('adding row: ' + str(row))
        # print(row)
        # print(format('\n'))
        test_array.append(x_data[row])
        y_test.append(y_data[row])

    return training, y_training, validation, y_validation, test_array, y_test, random_selection


'''
s_array = [.65, .20, .15]

train_number = np.around(398 * s_array[0])
val_number = np.around(398 * s_array[1])
test_number = np.around(398 * s_array[2])

test_number = 398 - (train_number + val_number)

print('train number')
print(train_number)
print('val number')
print(val_number)
print('test number')
print(test_number)
print('total')
print(train_number+val_number+test_number)
'''


# splits x data and y data into training and validation sets
# based on splitval array which holds percentages
def dos_data_splitter(xdata, ydata, splitval_array):
    training_data = []
    validation_data = []
    y_training = []
    y_validation = []

    train_limit = int(len(xdata)*splitval_array[0])
    #val_limit = len(xdata) - train_limit
    val_limit = len(xdata)

    # print(format('\n'))
    # print('train limit')
    # print(train_limit)
    # print('validation limit')
    # print(val_limit)
    # print(format('\n'))

    # used to pick random observations for each data set
    random_selection = np.random.choice(len(xdata), len(ydata), replace=False)

    # grab training data set
    for idx in range(0, train_limit):
        row = random_selection[idx]
        training_data.append(xdata[row])
        y_training.append(ydata[row])

    # grab validation data set
    for idx in range(train_limit, val_limit):
        row = random_selection[idx]
        validation_data.append(xdata[row])
        y_validation.append(ydata[row])
    return training_data, validation_data, y_training, y_validation, random_selection


def tres_data_splitter(xdata, ydata, splitval_array):
    training_data = list()
    validation_data = list()
    test_data = list()

    y_training = list()
    y_validation = list()
    y_test = list()

    train_limit = int(len(xdata)*splitval_array[0])
    val_limit = int(len(xdata)*splitval_array[1]) + train_limit
    test_limit = len(xdata)

    # print(format('\n'))
    # print('train limit')
    # print(train_limit)
    # print('validation limit')
    # print(val_limit)
    # print(format('\n'))

    # used to pick random observations for each data set
    random_selection = np.random.choice(len(xdata), len(ydata), replace=False)

    # grab training data set
    for idx in range(0, train_limit):
        row = random_selection[idx]
        training_data.append(xdata[row])
        y_training.append(ydata[row])
    # grab validation data set
    for idx in range(train_limit, val_limit):
        # for idx in range(train_limit, val_limit):
        row = random_selection[idx]
        validation_data.append(xdata[row])
        y_validation.append(ydata[row])
    for idx in range(val_limit, test_limit):
        row = random_selection[idx]
        test_data.append(xdata[row])
        y_test.append(ydata[row])
    return training_data, validation_data, y_training, y_validation, test_data, y_test, random_selection

# -----------------------------------------------------------------------------------------------------


# ----------------------------------------data manipulation--------------------------------------------
# gets the column col from the data array
def column_getter(array, col):
    y2d = [array[i][col:col+1] for i in range(0, len(array))]
    # print(format('\n'))
    # print('Y2d')
    # print(Y2d)
    # print(format('\n'))
    retlist = []
    for row in y2d:
        retlist.append(row[0])
    return retlist


# finds bad data points and returns a map
# keyed on the column and with a value of a list of the
# rows where the bad data is located
def find_col_bad_data(dataarray, badsig):
    retdic = {}
    for col in range(len(dataarray[0])):
            for row in range(len(dataarray)):
                if dataarray[row][col] == badsig:
                    if col in retdic:
                        retdic[col].append(row)
                    else:
                        rowlist = [row]
                        retdic[col] = rowlist
    return retdic


# removes selected data found in bad_list
def remove_row(array, bad_list):
    count = 0
    for entry in bad_list:
        del array[entry - count]
        count += 1
    return array


# removes the column col from the data array
def remove_col(array, col):
    for row in range(0, len(array)):
        del array[row][col]
    return array


# converts data array elements to either floats or ints depending on con_dis array
# will also replace unknown data signified with bad_sig with a zero
def convert_strings_float_int(data_array, bad_sig,  con_dis):
    for row in range(len(data_array)):
        # print(row)
        for col in range(len(data_array[0])):
            val = data_array[row][col]
            if val == bad_sig:
                data_array[row][col] = float(0)
                # del data_array[row]
                # count += 1
            else:
                if con_dis[col] == 1:
                    data_array[row][col] = int(data_array[row][col])
                else:
                    data_array[row][col] = float(data_array[row][col])

    return data_array


def getlinregmissingdata(regdata, baddic, w):
    r = []
    for entry in baddic:
        dlist = baddic[entry]
        for row in dlist:
            x = list()
            x.append(1)
            for col in range(len(regdata[0])):
                if col != entry:
                    x.append(regdata[row][col])
            xnp = np.array(x, dtype=np.float64)
            wnp = np.array(w, dtype=np.float64)
            r.append(np.dot(xnp, wnp))
    return r


# replaces bad data with the given value
def replace_bad_data_vec(data: list, baddic: dict, vec: list) -> list:
    for entry in baddic:
        badlist = baddic[entry]
        for idx in range(len(badlist)):
            row = badlist[idx]
            data[row][entry] = float(vec[idx])
    return data.copy()


# uses linear regression to fill in missing/bad data
def linear_regression_replacer(data, m, b, x_col, y_col, bad_data_points):
    datapointrows = bad_data_points[y_col]
    for row in datapointrows:
        x = data[row][x_col]
        print('x ', x)
        print(m*x+b)
        data[row][y_col] = m * x + b
    return data.copy()


def replace_item(data_array, bad_data_dictionary, value_array):

    count: int = 0
    for col in bad_data_dictionary:
        for row in bad_data_dictionary[col]:
            data_array[row][col] = value_array[count]
        count += 1
    return data_array


# returns an independent array made of the data minus y column and a dependent vector that comes from the column
# Y column out of the data array
# ignores the last column all together
def i_d_arrays(d_array, y_col):

    y = []
    x = []

    arry = list(d_array)

    for row in arry:

        y.append(row[y_col])

        # del row[Ycol]

        xc = list()
        xc.append(1)

        for idx in range(len(row)-1):
            if idx != y_col:
                xc.append(row[idx])

        x.append(xc)

    return y, x


# will return an X array full of independent variables, and a Y array of the dependant
# variables
def x_y_getter(array, y_col):
    y = column_getter(array, y_col)

    front = [array[i][0:y_col] for i in range(0, len(array))]
    back = [array[i][y_col + 1:] for i in range(0, len(array))]

    # chunk1 = front[0] + endr[0]
    # chunk2 = hlf1[1] + endr[1]
    # chunk = [chunk1] + [chunk2]

    x = [[1] + front[0] + back[0]]

    for idx in range(1, len(array)):
        x += [[1] + front[idx] + back[idx]]

    return x, y


def poly_x_maker(x_list, power):
    p_x = list()
    for row in range(len(x_list)):
        l = [1]
        p_x.append(l)
        for col in range(len(x_list[0])):
            for p in range(1, power + 1):
                p_x[row].append(np.power(x_list[row][col], p))
    return p_x


def get_col_avg(w_l, w_b_l):
    w_col_sum = list(w_l[w_b_l[0]])
    # list((map(operator.add, listsum, w_list[l])))
    div_list = [(len(w_b_l))]

    for idx in range(1, len(w_b_l)):
        w_col_sum = list(map(operator.add, w_col_sum, w_l[w_b_l[idx]]))
        div_list.append(len(w_b_l))

    avg_w = list(map(operator.truediv, w_col_sum, div_list))

    return avg_w


def get_fixed_data_stats_x_y_xn_yn(data_a, cont_dis, ind_col):
    # get x and y matrices
    x, y = x_y_getter(list(data_a), ind_col)

    smu, std, min_a, max_a = get_basic_stats(data_a, cont_dis)

    stat_array = [smu, std, min_a, max_a]

    ret_data = list()

    for row in data_a:
        ret_data.append(list(row))

    # print('data array in func')
    # print(data_a)

    orig_data = data_a[:]

    norm_dat = normalize_data(orig_data.copy(), smu, std, min_a, max_a, cont_dis)
    x_n, y_n = x_y_getter(list(norm_dat), ind_col)

    # print('orig array in func')
    # print(ret_data)
    # print('data array in func')
    # print(data_a)
    # print('normalized data array in func')
    # print(norm_dat)

    return ret_data, stat_array, x, y, x_n, y_n


def discard_imputation(data_array, cont_dis, cols_rmv, bad_sig, ind_col):
    baddatdic = find_col_bad_data(data_array.copy(), bad_sig)

    # remove the column for car name
    for col in cols_rmv:
        data_array = remove_col(list(data_array), col)


    # Convert strings to numerical values
    # using the continuous/discrete array to turn the value into a float or an int respectively
    data_array = convert_strings_float_int(list(data_array), bad_sig, cont_dis)
    orig_data = list(data_array)
    # remove the rows with bad data
    for entry in baddatdic:
        data_array = remove_row(list(data_array), baddatdic[entry])
    '''
    # get x and y matrices
    x, y = x_y_getter(list(data_array), ind_col)

    # grab the basic statistic for this data set
    smu, std, min_a, max_a = get_basic_stats(x, cont_dis)
    # smu_y, std_y, min_a_y, max_a_y = get_basic_stats(y, cont_dis)

    stat_array = [smu, std, min_a, max_a]
    ynp = np.array(y, dtype=float)

    smu_y = np.mean(ynp, dtype=float)
    std_y = np.std(ynp, dtype=float)
    min_a_y = np.amin(ynp)
    max_a_y = np.amax(ynp)

    stat_array_y = [smu_y, std_y, min_a_y, max_a_y]

    ret_data = list()

    for row in data_array:
        ret_data.append(list(row))



    print('data array in func')
    print(data_array)

    orig_data = data_array[:]

    x_norm = normalize_data(x, smu, std, min_a, max_a, cont_dis)
    y_norm = z_norm_col(y, smu_y, std_y)

    print('orig array in func')
    print(ret_data)
    print('data array in func')
    print(data_array)

    # x_norm = x_y_norm[0]
    # y_norm = x_y_norm[1]
    '''

    ret_data, stat_array, x, y, x_norm, y_norm = get_fixed_data_stats_x_y_xn_yn(data_array, cont_dis, ind_col)

    return ret_data, stat_array, x, y, x_norm, y_norm


def average_imputation(data_array, cont_dis, cols_rmv, bad_sig, ind_col):
    bad_dat_dic = find_col_bad_data(data_array.copy(), bad_sig)

    # print(bad_dat_dic)

    value_array = list()

    # remove the column for car name
    for col in cols_rmv:
        data_array = remove_col(list(data_array), col)

    data_array = convert_strings_float_int(list(data_array), bad_sig, cont_dis)

    dat_a = list(data_array)

    for row in bad_dat_dic:
        # calculate the average of the HorsePowerData
        # Get the horse power column
        #
        hp_array = column_getter(list(dat_a), row)

        # make a copy of it so we can average it up
        hp_averager = list(hp_array)

        # remove the rows with bad data points
        hp_averager = remove_row(hp_averager, bad_dat_dic[row])

        # calculate the average of the Horse Power attribute
        # and use it to replace missing data
        hp_mean = np.mean(np.array(hp_averager, dtype=np.float), dtype=np.float)

        value_array.append(hp_mean)

    # add the average in place of missing data
    data_array = replace_item(list(dat_a), bad_dat_dic, value_array)

    for avg in value_array:
        print('avg: ' + str(avg))

    # TODO: this is used for debugging and should be removed in the end
    # print(format('\n'))
    # for col in bad_dat_dic:
    #    for row in bad_dat_dic[col]:
    #        print('column' + str(col) + " row " + str(row))
    #        print(data_array[row][col])
    #        print(data_array[row])
    #print(format('\n'))

    # smu, std, min_a, max_a = get_basic_stats(data_array, cont_dis)

    # stat_array = [smu, std, min_a, max_a]

    # ret_data = list()

    # for row in data_array:
    #   ret_data.append(list(row))

    # ret_data, stat_array, x, y, x_norm, y_norm = get_fixed_data_stats_x_y_xn_yn(data_array, cont_dis, ind_col)

    return get_fixed_data_stats_x_y_xn_yn(data_array, cont_dis, ind_col)


def linear_regression_imputation(data_array, cont_dis, cols_rmv, bad_sig, ind_col):
    bad_dat_dic = find_col_bad_data(data_array.copy(), bad_sig)

    # print(bad_dat_dic)

    value_array = list()

    # remove the column for car name
    for col in cols_rmv:
        data_array = remove_col(list(data_array), col)

    data_array = convert_strings_float_int(list(data_array), bad_sig, cont_dis)

    dat_a = list(data_array)

    d_a_known = remove_row(list(dat_a), bad_dat_dic[3])

    x_a_lr_i, y_a_lr_i = x_y_getter(list(d_a_known), 3)

    '''
    print('x_a_lr')
    print(x_a_lr_i)
    print(len(x_a_lr_i))
    print('y_a_lr')
    print(y_a_lr_i)
    print(len(y_a_lr_i))
    '''

    min_col, min_mse, best_col = RegressionTools.find_first(list(x_a_lr_i), list(y_a_lr_i), list([.60, .40]))

    '''
    print('minimum col')
    print(min_col[0])
    print('minimuc mse')
    print(min_mse[0])
    print('best_col')
    print(best_col[0])
    '''


    # hp_array = DataProcessor.column_getter(list(y_a_lr_i), attribute_label_array.index('Horse Power'))
    hp_array = list(y_a_lr_i)
    weight_array = column_getter(list(x_a_lr_i), min_col[0])

    '''
    print(format('\n'))
    print('Horse Power column')
    print(hp_array)
    print(len(hp_array))
    print('Weight column')
    print(weight_array)
    print(len(weight_array))
    print(format('\n'))

    print('length of original data array')
    print(data_array)
    print(len(data_array))
    '''

    # m, b, x, y, yg, mse = RegressionTools.reg_lin_regression_msr(weight_array, hp_array, [.60, .40])

    # get a value for the parameter array W
    w_imp = RegressionTools.multi_linear_regressor(x_a_lr_i, hp_array)

    # use the parameter array to calculate values for the missing data points
    rt = getlinregmissingdata(list(data_array), bad_dat_dic, w_imp)

    # print('rt')
    # print(rt)

    imputation_data = replace_bad_data_vec(list(data_array), bad_dat_dic, rt)

    # print('l of imput')
    # print(len(imputation_data))
    # print(imputation_data[126])

    # smu, std, min_a, max_a = get_basic_stats(imputation_data, cont_dis)

    # stat_array = [smu, std, min_a, max_a]

    # ret_data = list()

    # for row in data_array:
    #     ret_data.append(list(row))

    # ret_data, stat_array, x, y, x_norm, y_norm = get_fixed_data_stats_x_y_xn_yn(data_array, cont_dis, ind_col)

    return get_fixed_data_stats_x_y_xn_yn(imputation_data, cont_dis, ind_col)

# -------------------------------------------------------------------------------------------------------
# ----------------------------------------Statistics----------------------------------------------------
def max_min_array_getter(attrib_data):
    min_array = []
    max_array = []

    # print(format('\n'))
    # print('len attrib data')
    # print(len(attrib_data))
    # print(format('\n'))

    for col in range(len(attrib_data[0])):
        # print(format('\n'))
        # print("col")
        # print(col)
        # print(format('\n'))
        attrib = np.array(column_getter(attrib_data, col), dtype=np.float)

        min_array.append(np.amin(attrib))
        max_array.append(np.amax(attrib))

    return min_array, max_array


def z_number(x, mu, sigma):
    return (x - mu)/sigma


def normalization(x, x_min, x_max):
    return (x - x_min)/(x_max - x_min)


def normalize_data(attrib_d, mean_l, sigma_l, min_l, max_l, con_dis):
    col_end = len(attrib_d[0])
    row_end = len(attrib_d)

    ret_array = list(attrib_d[:])

    for c in range(0, col_end):

        for r in range(0, row_end):

            if con_dis[c] == 0:
                ret_array[r][c] = z_number(attrib_d[r][c], mean_l[c], sigma_l[c])
            elif con_dis[c] == 1:
                ret_array[r][c] = normalization(attrib_d[r][c], min_l[c], max_l[c])

    return ret_array


def z_norm_col(y, mu, std):
    tmp_l = []
    for row in y:
        diffy = row - mu
        tmp_l.append(diffy/std)
    return tmp_l


# performs z normalization on the given attribute array
def z_normalization(attrib_data, mean_array, std_array):
    col_end = len(attrib_data[0])
    z_norm_vals = list(attrib_data)
    tmp_list = []

    for col in range(0, col_end):
        # attrib = np.array(GetColumn(attrib_data, col), dtype=np.float)
        attrib = np.array(column_getter(attrib_data, col), dtype=np.float)

        # print('attrib for column: ' + str(col))
        # print(attrib)
        tmp_list.clear()
        # sum = 0
        for row in range(len(attrib)):
            dif = (attrib[row] - mean_array[col])
            # print('dif at row: ' + str(row))
            # print(dif)
            # print('mean at col: ' + str(col) + ' is ')
            # print(mean_array[col])
            if std_array[col] == 0:
                tmp_list.append(dif/mean_array[col])
            else:
                tmp_list.append(dif/std_array[col])

        for row in range(len(z_norm_vals)):
            z_norm_vals[row][col] = tmp_list[row]

    return list(z_norm_vals)


# returns the sample mean of the attributes in the given array
# returning a int for discrete values and a float for continuous values
# uses the con_dis array to tell the difference
def sample_mean_array(array, con_dis):
    smu = []

    limit = len(array[0])

    for col in range(0, limit):
        if con_dis[col] == 0:
            smu.append(np.mean(np.array(column_getter(array, col), dtype=np.float), dtype=float))
        elif con_dis[col] == 1:
            smu.append(np.mean(np.array(column_getter(array, col), dtype=np.int), dtype=np.int))
    return smu


# returns the sample standard deveation of the attributes in the given array
def sample_std_array(attribdata):
    end = len(attribdata[0])
    ssig2 = []
    discrete_vals = [1, 6, 7]
    for col in range(0, end):
        if col in discrete_vals:
            attrib = np.array(column_getter(attribdata, col), dtype=np.int)
            ssig2.append(int(np.std(attrib, dtype=np.int)))
        else:
            attrib = np.array(column_getter(attribdata, col), dtype=np.float)
            ssig2.append((np.std(attrib, dtype=np.float)))
    return list(ssig2)


def quartiles_array(data):

    q_array = []

    for col in range(data[0]-1):
        attrib = column_getter(data, col)
        lq, uq, md, medn = quartiles(attrib)
        q_list = [lq, uq, md, medn]
        q_array.append(q_list)
    return q_array


def quartiles(attrib_data):
    sorted_points = sorted(attrib_data)
    # 2. divide the data set in two halves
    mid = len(sorted_points) / 2

    a_median: ndarray = np.median(attrib_data)

    if len(sorted_points) % 2 == 0:
        # even
        lower_q = np.median(sorted_points[:mid])
        upper_q = np.median(sorted_points[mid:])
    else:
        # odd
        lower_q = np.median(sorted_points[:mid])  # same as even
        upper_q = np.median(sorted_points[mid + 1:])

    return lower_q, upper_q, mid, a_median


# will calculate and return the basic stats of a data set
# i.e.
# sample mean
# sample standard deviation
# min attribute array
# max attribute array
def get_basic_stats(data_array, continuous_discrete):

    smu = sample_mean_array(data_array, continuous_discrete)

    std = sample_std_array(data_array)

    min_a, max_a = max_min_array_getter(data_array)

    return smu, std, min_a, max_a


# ------------------------------------------------------------------------------------------------------
