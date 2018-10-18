import numpy as np
import DataProcessor


# ---------------------------------------------Trainers---------------------------------------------------------
# used to attain the best test size with the highest coefficient of determination
# can use split_array to either run multiple runs of the same size array to get an average error
# or have differnt sizes in the array to look for the best size vs cod
def train_model_cod_dos(x_data, y_data, split_array):
    # training_data, validation_data, y_training, y_validation
    print(format("\n"))
    print(len(x_data))
    # split_array = int(len(x_data) / 16)
    best_cod = 0
    bs = list()
    cod_list = list()
    length_array = len(split_array)
    best_split = []
    best_rand = list(list([1]))
    rand_list = list()

    # while len(validation_set) >= 10:
    for x in range(0, length_array):
        train, validation, y_train, y_validation, rand = DataProcessor.dos_data_splitter(x_data, y_data, split_array[x])
        '''
        print('len(train_set)')
        print(len(train_set))
        print('length of validation_set')
        print(len(validation_set))
        print('length of y_training')
        print(len(y_training))
        print('length of y_validation')
        print(len(y_validation))
        '''
        w_params = multi_linear_regressor(train, y_train)
        # use w to get some response data
        gmodel = get_r_data(validation, w_params)
        cod = calculate_cod(gmodel, y_validation)
        cod_list.append(cod)
        rand_list.append(rand)
        if best_cod < cod and cod > 0 and cod <= 1:
            best_cod = cod
            bs = split_array[x]
            best_rand[0] = rand
    return best_cod, bs, cod_list, rand_list, best_rand


# used to attain the best test size with the hightest coefficient of determination
def train_model_lse_dos(x_data, y_data, split_array):
    # training_data, validation_data, y_training, y_validation
    # print(len(Xdata))
    print(format("\n"))
    bs = 10
    best_ls = 1000
    ls_list = list()
    rand_list = list()
    best_rand = list(list([0]))
    # train_set, validation_set, y_training, y_validation = DataProcessor.dos_data_splitter(Xdata, Ydata, split_array)
    # while len(validation_set) >= 10:
    for x in range(0, len(split_array)):
        '''
        print('len(train_set)')
        print(len(train_set))
        print('length of validation_set')
        print(len(validation_set))
        print('length of y_training')
        print(len(y_training))
        print('length of y_validation')
        print(len(y_validation))
        '''

        train, validation, y_training, y_validation, rand = DataProcessor.dos_data_splitter(x_data, y_data, split_array[x])
        w_params = multi_linear_regressor(train, y_training)
        # use w to get some response data
        gmodel = get_r_data(validation, w_params)
        ls = least_squares_estimate(gmodel, y_validation)
        ls_list.append(ls)
        rand_list.append(rand)
        if ls < best_ls:
            best_ls = ls
            bs = split_array[x]
            best_rand[0] = rand
    return best_ls, bs, ls_list, rand_list, best_rand


# used to attain the best test size with the highest coefficient of determination
def train_model_mse_dos(x_data, y_data, split_array):
    # training_data, validation_data, y_training, y_validation
    print(format("\n"))
    bs = 10
    best_mse = 1000
    mse_list = list()
    best_rand = list(list([0]))
    rand_list = list()

    for x in range(0, len(split_array)):
        train, validation, y_train, y_validation, rand = DataProcessor.dos_data_splitter(x_data, y_data, split_array[x])
        '''
        print('len(train)')
        print(len(train))
        print('length of validation_set')
        print(len(validation_set))
        print('length of y_training')
        print(len(y_training))
        print('length of y_validation')
        print(len(y_validation))
        '''
        w_params = multi_linear_regressor(train, y_train)
        # use w to get some response data
        g_model = get_r_data(validation, w_params)
        rand_list.append(rand)
        mse = mean_square_error(g_model, y_validation)
        mse_list.append(mse)
        rand_list.append(rand)
        if mse < best_mse:
            best_mse = mse
            bs = split_array[x]
            best_rand[0] = rand

    return best_mse, bs, mse_list, rand_list, best_rand,

# -------------------------------------------------------------------------------------------------------------

# ------------------------------------------------Error Functions----------------------------------------------


# calculates the Mean Square Error
def calculate_cod(g_model, r_validate):
    # get the mean of the validation data
    # print(format('\n'))
    # print("length of r validate")
    # print(len(r_validate))
    # print(format('\n'))
    r_mean = np.mean(r_validate)
    bottom = 0
    top = 0
    for idx in range(len(g_model)):
        top += np.power((r_validate[idx] - g_model[idx]), 2)
    for idx in range(len(r_validate)):
        bottom += np.power((r_validate[idx] - r_mean), 2)
    return 1 - top/bottom


# calculates the mean square error
def mean_square_error(d_array, yarray):
    n = len(d_array)
    difference_list = []
    for idx in range(n):
        diff = d_array[idx] - yarray[idx]
        #difference_list.append(pow(diff, 2))
        difference_list.append(np.absolute(diff))
    return np.mean(np.array(difference_list, dtype=np.float64))


# calculates the least squares estimate
def least_squares_estimate(d_array, y_array):
    n = len(d_array)
    summation = 0
    for idx in range(n):
        diff = d_array[idx] - y_array[idx]
        summation += abs(diff)
    return summation / 2


# --------------------------------------------------------------------------------------------------------------------

# -------------------------------------------Regression Functions------------------------------------------------------

def get_r_data(x_data, w):
    r = []

    # print('length of w')
    # print(len(w))
    # print(len([0]))
    # print('length x data')
    # print(len(x_data[0]))

    wnp = np.array(w, dtype=np.float64)
    for row in range(len(x_data)):
        x_observation = list()
        #x_observation.append(1)
        for col in range(len(x_data[0])):
            x_observation.append(x_data[row][col])
        r.append(np.dot(np.array(x_observation, dtype=np.float64), wnp))
    return r

# used for linear regression imputation
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
            for col in range(len(regdata[0])-1):
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


# performs multiple linear regrsion on the x and y data
# and returns the generated parameter vector W
def multi_linear_regressor(x_data, y_data):
    '''
    xnew = list()

    for r in range(len(x_data)):
        nlist = [1]
        for c in range(len(x_data[0])):
            nlist.append(x_data[r][c])
        xnew.append(nlist)
     '''


    x = np.array(x_data, dtype=np.float)
    y = np.array(y_data, dtype=np.float)
    x_transpose = np.transpose(x)
    xtx = np.dot(x_transpose, x)
    xtx_inv = np.linalg.inv(xtx)
    xtx_inv_xt = np.dot(xtx_inv, x_transpose)
    w = np.dot(xtx_inv_xt, y)
    return w


def linear_calculation_for_w(x, y):
    xsum = sum(x)
    ysum = sum(y)

    xy = [a * b for a, b in zip(x, y)]

    xx = [a * b for a, b in zip(x, x)]

    xxsum = sum(xx)

    xysum = sum(xy)

    n = len(x)

    a = [[n, xsum],
         [xsum, xxsum]]

    y = [ysum,
         xysum]

    anp = np.array(a)

    anpinv = np.linalg.inv(anp)

    ynp = np.array(y)

    w = np.dot(anpinv, ynp)

    return w


# uses linear regression to generate a slope(m) and intercept(b) value
# for a line approximating the data
# def reg_lin_regression_MSR(X, Y):
def reg_lin_regression_msr(x, y, split):
    # training_data, validation_data, y_training, y_validation
    train, validation, y_train, y_validation, rand = DataProcessor.dos_data_splitter(x, y, split)
    # train, validation, y_train, y_validation = data_splitter(X, Y, int(len(X)/16))

    w = linear_calculation_for_w(train, y_train)
    # b = W[0][0][0][0]
    # m = W[0][1][0][0]
    b = w[0]
    m = w[1]
    # print('b: ')
    # print(b)
    # print('m: ')
    # print(m)
    yg = list()
    for idx in range(len(validation)):
        val = m*validation[idx] + b
        yg.append(val)
    mse = mean_square_error(yg, y_validation)
    return m, b, x, y, yg, mse


# -----------------------------------------------------------------------------------------------------------------------


# --------------------------------------Dimensionality Reduction--------------------------------------------------------


# will use linear regression to find the first attribute
# to add to the F array for forward selection
# will split the data into training and validation sets
# according to split parameter
def find_first(x_data, y_data, split):
    col_size = len(x_data[0])
    min_mse = [10000]
    min_col = [10000]
    best_col = []
    for col in range(1,col_size):
        x_column = list(DataProcessor.column_getter(x_data, col))

        # print('x_column')
        # print(x_column)
        # print('col')
        # print(col)

        # training_data, validation_data, y_training, y_validation = data_splitter(x_column, y_data, len(x_column)/16)

        m, b, x, y, yg, mse = reg_lin_regression_msr(x_column, y_data, split)

        # print(format("\n"))
        # print('-------------------------mse')
        # print(mse)
        # print(format("\n"))
        # coef = [m,b]

        # y_result = GetYvals(coef, validation_data)
        # y_result = GetY

        if mse < min_mse[0]:
            min_col[0] = col
            min_mse[0] = mse
            best_col.clear()
            best_col.append(list(x_column))

    return list(min_col), list(min_mse), list(best_col)


# attempts to do forward selection
def forward_selector(x_data, y_data, split):

    nx = len(x_data)
    # print('X data is ')
    # print(x_data)
    # print(len(x_data))
    ny = len(y_data)
    col_size = len(x_data[0])
    used_col = []

    found = True

    addcol = [2000]

    mininmum_mse = [10000]

    F = list()
    Fsaver = list()

    # find the first variable  array to add to F as well its mean square error
    min_col, min_mse, best_col = find_first(x_data, y_data, split)

    print('min_col')
    print(min_col)
    print(format("\n"))

    print('min_mse')
    print(min_mse)
    print(format("\n"))

    print('best_col')
    print(best_col[0])
    print(len(best_col[0]))
    print(format("\n"))

    mininmum_mse[0] = min_mse

    # used to ignore column 1
    used_col.append(0)
    used_col.append(min_col[0])
    mininmum_mse[0] = min_mse[0]

    # set up F array
    for row in range(nx):
        flist = [1.0]
        F.append(flist)

    for row in range(nx):
        F[row].append(best_col[0][row])

    # print('in side funct F')
    # print(F)

    # while old_error > new_error:
    while found:
        found = False
        # go through all columns checking for the min mse value ans storeing that x column
        #   print(format("\n"))
        #  print('--------------------------------------------------------------------Starting for loop ')
        #  print('Length of F', len(F[0]))
        #  print(format("\n"))
        #  print(format("\n"))
        for col in range(1, col_size):
            if col not in used_col:
                # print('col')
                #    print(col)
                Ftmp = F[:]
                # Fsaver.count()

                # create a temp F array
                # each row of Ftmp contains a list
                # adds the current column
                for row in range(nx):
                    Ftmp[row].append(x_data[row][col])

                #       print('in for Ftmp')
                #      print(Ftmp)

                # split the data into training and validation sets
                train, validation, y_training, y_validation, rand = DataProcessor.dos_data_splitter(Ftmp, y_data, split)

                # perform linear regression to get W params
                w_params = multi_linear_regressor(train, y_training)

                # use w to get some response data
                gmodel = get_r_data(validation, w_params)

                # calculate the mean square error for this x column
                mse = mean_square_error(gmodel, y_validation)
                for row in range(nx):
                    Ftmp[row].pop()
                #     print('Ftmp is now')
                #    print(Ftmp)
                #   print(format("\n"))
                #   print('new mse')
                #   print(mse)

                if mse < mininmum_mse[0]:
                    #      print('found new min mse as '+ str(mse) + ' at col ' + str(col))
                    mininmum_mse[0] = mse
                    addcol[0] = col
                    found = True
                    # Fsaver = list(Ftmp)
                del Ftmp[col]
                # print('Ftmp is now')
                # print(Ftmp)
                # if col not in used_col:
                #   used_col.append(col)
            # else:
                # print('column ' + str(col) + ' is alread used')

                # x_add = GetColumn(x_data, addcol)
        # print(format("\n"))
        # print(format("\n"))
        # print('The for loop ended')
        # print(format("\n"))
        # print(format("\n"))
        if not found:
            #   print('no better mse was found')
            break
        else:
            # add the saved column to F
            #  print("found a new variable to add: " + str(addcol[0]))
            for row in range(nx):
                F[row].append(x_data[row][addcol[0]])
            used_col.append(addcol[0])
            if len(F) == col_size + 1:
                break
        # add best column
        # for row in range(nx):
        # flist.append(x_add[row])

        # for row in range(nx):
        #    F[row].append(flist)

    return F, mininmum_mse[0]