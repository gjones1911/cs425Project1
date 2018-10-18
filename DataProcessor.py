import numpy as np
# import pandas as pd
# import operator
# from DataCleaner import *

# local global variables
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
    val_limit = len(xdata) - train_limit

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
    for idx in range(train_limit, val_limit+train_limit):
        row = random_selection[idx]
        validation_data.append(xdata[row])
        y_validation.append(ydata[row])
    return training_data, validation_data, y_training, y_validation, random_selection

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

    #print('----------------------------------------------------------------regdata')
    #print(regdata)

    for entry in baddic:
        dlist = baddic[entry]
        for row in dlist:
            #print('------------row')
            #print(row)
            x = []
            x.append(1)
            for col in range(len(regdata[0])):
                if col != entry:
                    x.append(regdata[row][col])
            Xnp = np.array(x, dtype=np.float64)
            Wnp = np.array(w, dtype=np.float64)
            '''
            print('length of Wnp')
            print(Wnp)
            print('length Xnp')
            print(Xnp)
            '''
            r.append(np.dot(Xnp, Wnp))
    return r


# replaces bad data with the given value
def ReplaceBadDatavec(data, baddic, vec):
    for entry in baddic:
        badlist = baddic[entry]
        for idx in range(len(badlist)):
            row = badlist[idx]
            data[row][entry] = float(vec[idx])
    return data.copy()


# uses linear regression to fill in missing/bad data
def linear_regression_replacer(data, m, b, Xcol, Ycol, baddataPoints):
    datapointrows = baddataPoints[Ycol]
    for row in datapointrows:
        x = data[row][Xcol]
        print('x ',x)
        print(m*x+b)
        data[row][Ycol] = m*x + b
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

    ret_array = attrib_d[:]

    for c in range(0, col_end):

        for r in range(0, row_end):

            if con_dis[c] == 0:
                ret_array[r][c] = z_number(attrib_d[r][c], mean_l[c], sigma_l[c])
            elif con_dis[c] == 1:
                ret_array[r][c] = normalization(attrib_d[r][c], min_l[c], max_l[c])

    return ret_array


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
# ------------------------------------------------------------------------------------------------------
