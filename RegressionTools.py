import numpy as np
import DataProcessor


# ---------------------------------------------Trainers---------------------------------------------------------
# used to attain the best test size with the highest coefficient of determination
def train_model_cod_dos(x_data, y_data, split_array):
    # training_data, validation_data, y_training, y_validation
    print(format("\n"))
    print(len(x_data))
    # split_array = int(len(x_data) / 16)
    best_cod = 0
    bs = list()
    best_ls = 1000
    best_mse = 1000
    length_array = len(split_array)

    # while len(validation_set) >= 10:
    for x in range(0, length_array):
        train_set,validation_set,y_training,y_validation = DataProcessor.dos_data_splitter(x_data,y_data,split_array[x])
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
        w_params = mulit_linear_regressor(train_set, y_training)
        # use w to get some response data
        gmodel = get_r_data(validation_set, w_params)
        cod = calculate_mse(gmodel, y_validation)
        if best_cod < cod and cod >  0 and cod <= 1:
            best_cod = cod
            bs = split_array[x]
    return best_cod, bs


# used to attain the best test size with the hightest coefficient of determination
def train_model_lse_dos(Xdata, Ydata, split_array):
    #training_data, validation_data, y_training, y_validation
    #print(len(Xdata))
    print(format("\n"))
    inc = 1
    best_cod = 0
    bs = 10
    best_ls = 1000
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

        train_set,validation_set,y_training,y_validation = DataProcessor.dos_data_splitter(Xdata, Ydata, split_array[x])
        w_params = mulit_linear_regressor(train_set, y_training)
        # use w to get some response data
        gmodel = get_r_data(validation_set, w_params)
        ls = least_squares_estimate(gmodel, y_validation)
        if ls < best_ls:
            best_ls = ls
            bs = split_array[x]
    return best_ls, bs


# used to attain the best test size with the hightest coefficient of determination
def train_model_mse_dos(Xdata, Ydata, split_array):
    #training_data, validation_data, y_training, y_validation
    print(format("\n"))

    N = len(Xdata)

    print(N)
    split_array = int(N / 16)
    inc = 1
    bs = 10

    best_mse = 1000

    #while len(validation_set) >= 10:
    for x in range(0, len(split_array)):
        train_set, validation_set, y_training, y_validation = DataProcessor.data_splitter(Xdata, Ydata, split_array[x])
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
        w_params = mulit_linear_regressor(train_set, y_training)
        # use w to get some response data
        gmodel = get_r_data(validation_set, w_params)

        mse = mean_square_error(gmodel, y_validation)
        if mse < best_mse:
            best_mse = mse
            bs = split_array

    return best_mse, bs

# -------------------------------------------------------------------------------------------------------------

# ------------------------------------------------Error Functions----------------------------------------------


# calculates the Mean Square Error
def calculate_mse(Gmodel, Rvalidate):
    #get the mean of the validation data
    r_mean = np.mean(Rvalidate)
    bottom = 0
    top = 0
    for idx in range(len(Gmodel)):
        top += np.power((Rvalidate[idx] - Gmodel[idx]), 2)
    for idx in range(len(Rvalidate)):
        bottom += np.power((Rvalidate[idx] - r_mean), 2)
    return 1 - top/bottom


# calculates the mean square error
def mean_square_error(d_array, yarray):
    N = len(d_array)
    diffList = []
    for idx in range(N):
        diff = d_array[idx] - yarray[idx]
        diffList.append(pow(diff, 2))
    return np.mean(np.array(diffList, dtype=np.float64))


# calculates the least squares estimate
def least_squares_estimate(d_array, y_array):
    N = len(d_array)
    summation = 0
    for idx in range(N):
        diff = d_array[idx] - y_array[idx]
        summation += abs(diff)
    return summation / 2


# --------------------------------------------------------------------------------------------------------------------

# -------------------------------------------Regression Funtions------------------------------------------------------

def get_r_data(x_data, w):
    r = []
    wnp = np.array(w, dtype=np.float64)
    for row in range(len(x_data)):
        x_observation = list()
        x_observation.append(1)
        for col in range(len(x_data[0])):
            x_observation.append(x_data[row][col])
        r.append(np.dot(np.array(x_observation, dtype=np.float64), wnp))
    return r


# performs multiple linear regrsion on the x and y data
# and returns the generated parameter vector W
def mulit_linear_regressor(x_data, y_data):
    x = np.array(x_data, dtype=np.float)
    y = np.array(y_data, dtype=np.float)
    x_transpose = np.transpose(x)
    xtx = np.dot(x_transpose, x)
    xtx_inv = np.linalg.inv(xtx)
    xtx_inv_xt = np.dot(xtx_inv, x_transpose)
    w = np.dot(xtx_inv_xt, y)
    return w


def linear_calculation_for_w(x, y):
    Xsum = sum(x)
    Ysum = sum(y)

    XY = [a * b for a, b in zip(x, y)]

    XX = [a * b for a, b in zip(x, x)]

    xxsum = sum(XX)

    xysum = sum(XY)

    n = len(x)

    a = [[n, Xsum],
         [Xsum, xxsum]]

    y = [Ysum,
         xysum]

    anp = np.array(a)

    anpinv = np.linalg.inv(anp)

    ynp = np.array(y)

    w = np.dot(anpinv, ynp)

    return w



# uses linear regression to generate a slpme(m) and intercept(b) value
# for a line approximating the data
#def reg_lin_regression_MSR(X, Y):
def reg_lin_regression_MSR(X, Y, split):

    #training_data, validation_data, y_training, y_validation

    train, validation, y_train, y_validation = DataProcessor.dos_data_splitter(X, Y, split)
    #train, validation, y_train, y_validation = data_splitter(X, Y, int(len(X)/16))


    W = linear_calculation_for_w(train, y_train)

    #b = W[0][0][0][0]
    #m = W[0][1][0][0]

    b = W[0]
    m = W[1]


    #print('b: ')
    #print(b)
    #print('m: ')
    #print(m)

    Yg = []

    for idx in range(len(validation)):
        val = m*validation[idx] + b
        Yg.append(val)

    mse = mean_square_error(Yg, y_validation)

    return m, b, X, Y, Yg, mse


# -----------------------------------------------------------------------------------------------------------------------

