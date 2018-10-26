import numpy as np
import DataProcessor
import operator


# -----------------------------------Teasting for error functions-------------------------------------------- #

# -----------------------------------Least Squares Estimate---------------------------------------------------
# will perform linear regression using the w list(w_l)
# and the data list(dat_l) to get some model values
# and then calculate the various lse's using those model values and
# the y data list(y_dat_l)
# returns an array containing:
# a list of all calculated lse values(dat_lse), a list of the best lse's (dat_best))
# The best lse found during testing (best_lse), and the index into the W list where
# the generated the best lse's
def test_data_set_lse(w_l, dat_l, y_dat_l):
    best_lse = 10000        # used to keep track of smalles LSE
    dat_lse = list()     # a list of all calculated lse
    dat_best = list()    # a list of the best lse
    best_w_idx = list()  # the indices into w_1 array that got the best lse

    for i in range(len(w_l)):
        gmodel = get_r_data(dat_l[i], w_l[i])
        lse = np.around(least_squares_estimate(gmodel, y_dat_l[i]), 4)
        dat_lse.append(lse)
        if best_lse > lse:
            dat_best.append(lse)
            best_lse = lse
            best_w_idx.append(i)
    ret_array = [dat_lse, dat_best, best_lse, best_w_idx]

    return ret_array


def test_data_set_lse_b(w_l, dat_l, y_dat_l):
    dat_lse = list()
    test_dic = {}
    avg_lse = [0]
    test_num = 20
    best_avg = 10000

    for i in range(len(w_l)):

        lse_l = list()
        # test the randomized sameple
        for test in range(test_num):
            dat_r_s = list()
            y_rs = list()
            r_select = np.random.choice(len(dat_l[i]), len(y_dat_l[i]), replace=False)

            # get a randomized form of the test sample
            for r_s in r_select:
                dat_r_s.append(dat_l[i][r_s])
                y_rs.append(y_dat_l[i][r_s])

            gmodel = get_r_data(dat_r_s, w_l[i])
            lse = np.around(least_squares_estimate(gmodel, y_rs), 4)
            dat_lse.append(lse)

            lse_l.append(lse)

        avg_lse[0] = np.mean(lse_l, dtype=np.float)

        if best_avg > avg_lse[0]:
            best_avg = avg_lse[0]
            test_dic[i] = np.around(avg_lse, 4)

    ret_tuple = sorted(test_dic.items(), key=operator.itemgetter(1), reverse=False)

    return ret_tuple


# -----------------------------------COD-------------------------------------------------------------
# will perform linear regression using the w list(w_l)
# and the data list(dat_l) to get some model values
# and then calculate the various COD's using those model values and
# the y data list(y_dat_l)
# returns an array containing:
# an list of all calculated cod values(dat_cod), a list of the best cod's (dat_best))
# The best COD found during testing (best_cod), and the index into the W list where
# the generated the best COD's
def test_data_set_cod(w_l, dat_l, y_dat_l):

    best_cod = 0
    dat_cod = list()
    dat_best = list()
    best_w_idx = list()

    avg_cod = list()

    for i in range(len(w_l)):
        gmodel = get_r_data(dat_l[i], w_l[i])
        cod = np.around(calculate_cod(gmodel, y_dat_l[i]), 4)
        dat_cod.append(cod)

        if best_cod < cod and cod > 0 and cod <= 1:
            dat_best.append(cod)
            best_cod = cod
            best_w_idx.append(i)

    ret_array = [dat_cod, dat_best, best_cod, best_w_idx]

    return ret_array


def test_cod(w, dat_l, y_dat_l, idx, dic):

    best_cod = 0
    dat_cod = list()
    dat_best = list()
    best_w_idx = list()

    avg_cod = list()

    for i in range(len(dat_l)):
        gmodel = get_r_data(dat_l[i], w)
        cod = np.around(calculate_cod(gmodel, y_dat_l[i]), 4)
        dat_cod.append(cod)

        if best_cod < cod and cod > 0 and cod <= 1:
            dat_best.append(cod)
            best_cod = cod
            best_w_idx.append(i)

    ret_avg = np.around(np.mean(np.array(dat_cod, dtype=np.float), dtype=np.float), dtype=np.float)

    dic[idx] = ret_avg

    return dic


def test_data_set_cod_b(w_l, dat_l, y_dat_l):

    best_cod = 0
    dat_cod = list()
    dat_best = list()
    best_w_idx = list()
    test_dic = {}
    avg_cod = [0]
    test_num = 20
    best_avg = 0

    for i in range(len(w_l)):

        cod_l = list()
        # test the randomized sameple
        for test in range(test_num):
            dat_r_s = list()
            y_rs = list()
            r_select = np.random.choice(len(dat_l[i]), len(y_dat_l[i]), replace=False)

            # get a randomized form of the test sample
            for r_s in r_select:
                dat_r_s.append(dat_l[i][r_s])
                y_rs.append(y_dat_l[i][r_s])

            gmodel = get_r_data(dat_r_s, w_l[i])
            cod = np.around(calculate_cod(gmodel, y_rs), 4)
            dat_cod.append(cod)

            cod_l.append(cod)

        avg_cod[0] = np.mean(cod_l, dtype=np.float)

        if best_avg < avg_cod[0] and avg_cod[0] > 0 and avg_cod[0] <= 1:
            best_avg = avg_cod[0]
            test_dic[i] = np.around(avg_cod, 4)

    ret_tuple = sorted(test_dic.items(), key = operator.itemgetter(1), reverse=True)

    return ret_tuple


# Runs tests on all given data (dat_l) using the parameters (w_l)
# and uses the dependent data (y_dat_l) to calculate the COD
# for each parameter and storing the average COD for that parameter
# Returns: tet_tuple:  a list of tuples containing the lndex in w_l, and the the best average ocd generated by that w
#          W_tuple: a list of tuples containing the lndex in w_l, and the the best average ocd generated by that w
#          w_ret_l: a list containing the top 20 w's that had the best Cod, will be used to validation test
#          sorted_cods: a sorted version of all cod's generated in the order of largest to smallest
#          cods_best: a list of the best  avg cod's
#          the_best_cod: the highest cod generated
def test_data_set_cod_a(w_l, dat_l, y_dat_l):

    dat_cod = list()
    cods_best = list()
    test_dic = {}
    ret_dic = {}
    avg_cod = [0]
    best_avg = 0
    w_ret_l = []
    cod_l = list()

    for i in range(len(w_l)):
        current_cod_l = list()
        # test the randomized sameple
        for idx in range(len(dat_l)):
            gmodel = get_r_data(dat_l[idx], w_l[i])
            cod = np.around(calculate_cod(gmodel, y_dat_l[idx]), 12)
            current_cod_l.append(cod)

            cod_l.append(cod)

        # store the average cod for current W
        avg_cod[0] = np.mean(current_cod_l, dtype=np.float)

        # if the current average is better than the stored best store it
        if best_avg < avg_cod[0] and avg_cod[0] > 0 and avg_cod[0] <= 1:
            cods_best.append(cod)
            best_avg = avg_cod[0]
            test_dic[i] = np.around(avg_cod[0], 12)

        ret_dic[i] = np.around(best_avg, 12)

    sorted_cods = sorted(cod_l, reverse=True)

    the_best_cod = sorted_cods[0]

    ret_tuple = sorted(test_dic.items(), key=operator.itemgetter(1), reverse=True)
    w_tuple = sorted(ret_dic.items(), key=operator.itemgetter(1), reverse=True)

    for idx in range(20):
        w_ret_l.append(w_l[w_tuple[idx][0]])

    return ret_tuple, w_tuple, w_ret_l, sorted_cods, sorted(cods_best, reverse=True)[:10], the_best_cod


# Runs tests on all given data (dat_l) using the parameters (w_l)
# and uses the dependent data (y_dat_l) to calculate the COD
# for each parameter and storing the average COD for that parameter
# Returns: tet_tuple:  a list of tuples containing the lndex in w_l, and the the best average ocd generated by that w
#          W_tuple: a list of tuples containing the lndex in w_l, and the the best average ocd generated by that w
#          w_ret_l: a list containing the top 20 w's that had the best Cod, will be used to validation test
#          sorted_cods: a sorted version of all cod's generated in the order of largest to smallest
#          cods_best: a list of the best cods
#          the_best_cod: the highest cod generated
def test_data_set_lse_a(w_l, dat_l, y_dat_l):

    dat_lse = list()
    lse_best = list()
    test_dic = {}
    ret_dic = {}
    avg_lse = [0]
    best_avg = 10000
    w_ret_l = []
    lse_l = list()

    for i in range(len(w_l)):

        # test the randomized sameple
        for idx  in range(len(dat_l)):
            gmodel = get_r_data(dat_l[idx], w_l[i])
            lse = np.around(least_squares_estimate(gmodel, y_dat_l[idx]), 4)
            dat_lse.append(lse)

            lse_l.append(lse)

        avg_lse[0] = np.mean(lse_l, dtype=np.float)

        if best_avg > avg_lse[0]:
            lse_best.append(lse)
            best_avg = avg_lse[0]
            test_dic[i] = np.around(avg_lse[0], 4)

        ret_dic[i] = np.around(avg_lse[0], 4)

    sorted_lses = sorted(lse_l, reverse=False)

    the_best_lse = sorted_lses[0]

    ret_tuple = sorted(test_dic.items(), key=operator.itemgetter(1), reverse=False)
    w_tuple = sorted(ret_dic.items(), key=operator.itemgetter(1), reverse=False)

    for idx in range(20):
        w_ret_l.append(w_l[w_tuple[idx][0]])

    return ret_tuple, w_tuple, w_ret_l, sorted_lses, sorted(lse_best, reverse=False), the_best_lse


# Runs tests on all given data (dat_l) using the parameters (w_l)
# and uses the dependent data (y_dat_l) to calculate the Mean Square Error (mse)
# for each parameter and storing the average Lse for that parameter
# Returns: tet_tuple:  a list of tuples containing the lndex in w_l, and the the best average mse generated by that w
#          W_tuple: a list of tuples containing the lndex in w_l, and the the best average mse generated by that w
#          w_ret_l: a list containing the top 20 w's that had the best mse, will be used to validation test
#          sorted_mses: a sorted version of all mse's generated in the order of largest to smallest
#          cods_best: a list of the best mses
#          the_best_mse: the highest cod generated
def test_data_set_mse_a(w_l, dat_l, y_dat_l):

    dat_mse = list()
    mse_best = list()
    test_dic = {}
    ret_dic = {}
    avg_mse = [0]
    best_avg = 1000
    w_ret_l = []
    mse_l = list()

    for i in range(len(w_l)):

        # test the randomized sameple
        for idx  in range(len(dat_l)):
            gmodel = get_r_data(dat_l[idx], w_l[i])
            mse = np.around(mean_square_error(gmodel, y_dat_l[idx]), 4)
            dat_mse.append(mse)

            mse_l.append(mse)

        avg_mse[0] = np.mean(mse_l, dtype=np.float)

        if best_avg > avg_mse[0]:
            mse_best.append(mse)
            best_avg = avg_mse[0]
            test_dic[i] = np.around(avg_mse[0], 4)

        ret_dic[i] = np.around(avg_mse[0], 4)

    sorted_mses = sorted(mse_l, reverse=False)

    the_best_mse = sorted_mses[0]

    ret_tuple = sorted(test_dic.items(), key=operator.itemgetter(1), reverse=False)
    w_tuple = sorted(ret_dic.items(), key=operator.itemgetter(1), reverse=False)

    for idx in range(20):
        w_ret_l.append(w_l[w_tuple[idx][0]])

    return ret_tuple, w_tuple, w_ret_l, sorted_mses, sorted(mse_best, reverse=False), the_best_mse




# will perform linear regression using the w list(w_l)
# and the data list(dat_l) to get some model values
# and then calculate the various COD's using those model values and
# the y data list(y_dat_l)
# returns an array containing:
# an list of all calculated cod values(dat_cod), a list of the best cod's (dat_best))
# The best COD found during testing (best_cod), and the index into the W list where
# the generated the best COD's
def test_data_set_mse(w_l, dat_l, y_dat_l):

    best_mse = 1000
    dat_mse = list()
    dat_best = list()
    best_w_idx = list()

    for i in range(len(w_l)):
        gmodel = get_r_data(dat_l[i], w_l[i])
        mse = np.around(mean_square_error(gmodel, y_dat_l[i]), 4)
        dat_mse.append(mse)

        if best_mse > mse:
            dat_best.append(mse)
            best_mse = mse
            best_w_idx.append(i)

    ret_array = [dat_mse, dat_best, best_mse, best_w_idx]

    return ret_array


def test_data_set_mse_b(w_l, dat_l, y_dat_l):
    best_lse = 10000  # used to keep track of smalles LSE
    dat_mse = list()
    test_dic = {}
    avg_mse = [0]
    test_num = 20
    best_avg = 10000

    for i in range(len(w_l)):

        mse_l = list()
        # test the randomized sameple
        for test in range(test_num):
            dat_r_s = list()
            y_rs = list()
            r_select = np.random.choice(len(dat_l[i]), len(y_dat_l[i]), replace=False)

            # get a randomized form of the test sample
            for r_s in r_select:
                dat_r_s.append(dat_l[i][r_s])
                y_rs.append(y_dat_l[i][r_s])

            gmodel = get_r_data(dat_r_s, w_l[i])
            mse = np.around(mean_square_error(gmodel, y_rs), 4)
            dat_mse.append(mse)

            mse_l.append(mse)

        avg_mse[0] = np.mean(mse_l, dtype=np.float)

        if best_avg > avg_mse[0]:
            best_avg = avg_mse[0]
            test_dic[i] = np.around(avg_mse, 4)

    ret_tuple = sorted(test_dic.items(), key=operator.itemgetter(1), reverse=False)

    return ret_tuple

# -----------------------------------------------Collect parameters-----------------------------------------------


# Splits the given data into training and validation sets
# uses the test set to calculate a list of W values (parameters w_list)
# a list of training sets (tr_l), y values for that training set (y_tr_l)
# a list of validation sets, and there y values (val_l and y_val_l respectively)
# returns those in a list in the order: [w_list, tr_l, y_tr_list, val_l, y_val_l]]
#                                          0      1       2         3       4
def collect_parameters2(x_d, y_d, split_a):
    w_list = list()
    tr_l = list()
    y_tr_l = list()
    val_l = list()
    y_val_l = list()

    print('\n')

    for x in range(len(split_a)):
        tr, val, y_tr, y_val, rand = DataProcessor.dos_data_splitter(x_d, y_d, split_a[x])

        # get w from training data
        w_list.append(multi_linear_regressor(tr, y_tr))
        tr_l.append(tr)
        y_tr_l.append(y_tr)
        val_l.append(val)
        y_val_l.append(y_val)

    print('\n')
    return w_list, tr_l, y_tr_l, val_l, y_val_l


# ------------------------------------Used to get the average of the best training and validation W's----------------

def get_avg_best_w(w_l, best_train_w, best_val_w, tr_l, y_tr_l, val_l, y_val_l):

    # grab the largest cod for each training and validation sets
    tr_b_w = best_train_w
    val_b_w = best_val_w

    # get the average between the best cod for training and validation best W
    w_col_sum = list(map(operator.add, tr_b_w, val_b_w))
    div_l = list()
    for x in range(len(w_col_sum)):
        div_l.append(2)

    avg_all_w = list(map(operator.truediv, w_col_sum, div_l))

    tr_avg_cod = list()
    val_avg_cod = list()

    for x in range(len(tr_l)):
        g_t = get_r_data(tr_l[x], avg_all_w)
        cod_a = calculate_cod(g_t, y_tr_l[x])
        tr_avg_cod.append(cod_a)

    for x in range(len(val_l)):
        g_t = get_r_data(val_l[x], avg_all_w)
        cod_a = calculate_cod(g_t, y_val_l[x])
        val_avg_cod.append(cod_a)

    val_avg_cod_avg_b = np.mean(np.array(val_avg_cod, dtype=np.float), dtype=np.float)
    tr_avg_cod_avg_b = np.mean(np.array(tr_avg_cod, dtype=np.float), dtype=np.float)

    return val_avg_cod_avg_b, tr_avg_cod_avg_b


def get_avg_best_w_lse(w_l, best_train_w, best_val_w, tr_l, y_tr_l, val_l, y_val_l):

    # grab the largest cod for each training and validation sets
    tr_b_w = best_train_w
    val_b_w = best_val_w

    # get the average between the best cod for training and validation best W
    w_col_sum = list(map(operator.add, tr_b_w, val_b_w))
    div_l = list()
    for x in range(len(w_col_sum)):
        div_l.append(2)

    avg_all_w = list(map(operator.truediv, w_col_sum, div_l))

    tr_avg_cod = list()
    val_avg_cod = list()

    for x in range(len(tr_l)):
        g_t = get_r_data(tr_l[x], avg_all_w)
        lse_a = least_squares_estimate(g_t, y_tr_l[x])
        tr_avg_cod.append(lse_a)

    for x in range(len(val_l)):
        g_t = get_r_data(val_l[x], avg_all_w)
        lse_a = least_squares_estimate(g_t, y_val_l[x])
        val_avg_cod.append(lse_a)

    val_avg_cod_avg_b = np.mean(np.array(val_avg_cod, dtype=np.float), dtype=np.float)
    tr_avg_cod_avg_b = np.mean(np.array(tr_avg_cod, dtype=np.float), dtype=np.float)

    return val_avg_cod_avg_b, tr_avg_cod_avg_b


def get_avg_best_w_mse(w_l, best_train_w, best_val_w, tr_l, y_tr_l, val_l, y_val_l):

    # grab the largest cod for each training and validation sets
    tr_b_w = best_train_w
    val_b_w = best_val_w

    # get the average between the best cod for training and validation best W
    w_col_sum = list(map(operator.add, tr_b_w, val_b_w))
    div_l = list()
    for x in range(len(w_col_sum)):
        div_l.append(2)

    avg_all_w = list(map(operator.truediv, w_col_sum, div_l))

    tr_avg_cod = list()
    val_avg_cod = list()

    for x in range(len(tr_l)):
        g_t = get_r_data(tr_l[x], avg_all_w)
        mse_a = mean_square_error(g_t, y_tr_l[x])
        tr_avg_cod.append(mse_a)

    for x in range(len(val_l)):
        g_t = get_r_data(val_l[x], avg_all_w)
        mse_a = mean_square_error(g_t, y_val_l[x])
        val_avg_cod.append(mse_a)

    val_avg_cod_avg_b = np.mean(np.array(val_avg_cod, dtype=np.float), dtype=np.float)
    tr_avg_cod_avg_b = np.mean(np.array(tr_avg_cod, dtype=np.float), dtype=np.float)

    return val_avg_cod_avg_b, tr_avg_cod_avg_b


# ---------------------------------------------------------------------------------------------------------------
# will attempt to train the data using the objects in
# a list of parameters:
#  values(w), training data, y for training data, validation data, y for the validation data
def train_model_cod2(param_tr_val_a):
    # a list of parmeter vectors
    w_list = param_tr_val_a[0]

    # a list of training data sets
    tr_l = param_tr_val_a[1]

    # the dependent variables of the training sets
    y_tr_l = param_tr_val_a[2]

    # a list of validation data sets
    val_l = param_tr_val_a[3]

    # the dependent variables of the training sets
    y_val_l = param_tr_val_a[4]

    # ret_tuple,    w_tuple,     w_ret_l, sorted cods,  all the best cods, the best cod
    ret_tuple_trn, w_tuple_trn, tr_w_l, cod_tr, cods_best_tr, t_b_cod_t = test_data_set_cod_a(w_list, tr_l, y_tr_l)
    ret_tuple_val, w_tuple_val, val_w_l, cod_val, cods_best_val, t_b_cod_v= test_data_set_cod_a(tr_w_l, val_l, y_val_l)

    # tr_cod, tr_best, best_codtr, best_w_idx_tr = test_data_set_cod(w_list, tr_l, y_tr_l)
    # val_cod, val_best, best_codval, best_w_idx_val = test_data_set_cod(w_list, val_l, y_val_l)

    #ret_tuple_trn = test_data_set_cod_b(w_list, tr_l, y_tr_l)
    #ret_tuple_val = test_data_set_cod_b(w_list, val_l, y_val_l)

    print(format('\n'))
    print("Training set")
    print('ret tuple trn')
    print(ret_tuple_trn)
    print(ret_tuple_trn[0][0])
    print(ret_tuple_trn[0][1])
    print('w_tuple trn')
    print(w_tuple_trn)
    print(w_tuple_trn[0][0])
    print(w_tuple_trn[0][1])
    print(format('\n'))

    print("Validation set")
    print('ret tuple val')
    print(ret_tuple_val)
    print(ret_tuple_val[0][0])
    print(ret_tuple_val[0][1])
    print('w tuple val')
    print(w_tuple_val)
    print(w_tuple_val[0][0])
    print(w_tuple_val[0][1])
    print(format('\n'))


    # grab the largest cod for each training and validation sets
    # tr_b_w = w_list[best_w_idx_tr[-1]]
    # val_b_w = w_list[best_w_idx_val[-1]]

    tr_b_w = w_list[int(ret_tuple_trn[0][0])]
    val_b_w = w_list[int(ret_tuple_val[0][0])]

    val_avg_cod_avg_b, tr_avg_cod_avg_b = get_avg_best_w(w_list, tr_b_w, val_b_w, tr_l, y_tr_l, val_l, y_val_l)
    tr_avg_cod = np.mean(np.array(cod_tr, dtype=np.float), dtype=np.float)
    val_avg_cod = np.mean(np.array(cod_val, dtype=np.float), dtype=np.float)

    # ret_val = [[tr_cod, tr_best, best_codtr, best_w_idx_tr, tr_avg_cod],
    #          [val_cod, val_best, best_codval, best_w_idx_val, val_avg_cod],
    #           [tr_avg_cod_avg_b, val_avg_cod_avg_b]]

    ret_val = [[cod_tr, cods_best_tr, t_b_cod_t, ret_tuple_trn[0][0], tr_avg_cod],
               [cod_val, cods_best_val, t_b_cod_v,ret_tuple_val[0][0], val_avg_cod],
               [tr_avg_cod_avg_b, val_avg_cod_avg_b]]

    return ret_val


# will attempt to train the data using the objects in
# a list of parameter values(w), training data, y for training data
# validation data, y for the validation data
def train_model_lse2(param_tr_val_a):
    w_list = param_tr_val_a[0]

    tr_l = param_tr_val_a[1]
    y_tr_l = param_tr_val_a[2]

    val_l = param_tr_val_a[3]
    y_val_l = param_tr_val_a[4]

    # ret_tuple,    w_tuple,     w_ret_l, sorted cods,  all the best cods, the best cod
    ret_tuple_trn, w_tuple_trn, tr_w_l, lse_tr, lses_best_tr, t_b_lse_t = test_data_set_lse_a(w_list, tr_l, y_tr_l)
    ret_tuple_val, w_tuple_val, val_w_l, lse_val, lses_best_val, t_b_lse_v = test_data_set_lse_a(tr_w_l, val_l, y_val_l)

    # tr_cod, tr_best, best_codtr, best_w_idx_tr = test_data_set_cod(w_list, tr_l, y_tr_l)
    # val_cod, val_best, best_codval, best_w_idx_val = test_data_set_cod(w_list, val_l, y_val_l)

    # ret_tuple_trn = test_data_set_cod_b(w_list, tr_l, y_tr_l)
    # ret_tuple_val = test_data_set_cod_b(w_list, val_l, y_val_l)

    print(' sorted best lse across all w\'s')
    print(lses_best_tr)



    print(format('\n'))
    print("Training set")
    print('ret tuple')
    print(ret_tuple_trn)
    print(ret_tuple_trn[0][0])
    print(ret_tuple_trn[0][1])
    print(w_tuple_trn)
    print(w_tuple_trn[0][0])
    print(w_tuple_trn[0][1])
    print(format('\n'))

    print("Validation set")
    print('ret tuple')
    print(ret_tuple_val)
    print(ret_tuple_val[0][0])
    print(ret_tuple_val[0][1])
    print(w_tuple_val)
    print(w_tuple_val[0][0])
    print(w_tuple_val[0][1])
    print(format('\n'))

    # grab the largest cod for each training and validation sets
    # tr_b_w = w_list[best_w_idx_tr[-1]]
    # val_b_w = w_list[best_w_idx_val[-1]]

    tr_b_w = w_list[int(ret_tuple_trn[0][0])]
    val_b_w = w_list[int(ret_tuple_val[0][0])]

    val_avg_lse_avg_b, tr_avg_lse_avg_b = get_avg_best_w_lse(w_list, tr_b_w, val_b_w, tr_l, y_tr_l, val_l, y_val_l)
    tr_avg_lse = np.mean(np.array(lse_tr, dtype=np.float), dtype=np.float)
    val_avg_lse = np.mean(np.array(lse_val, dtype=np.float), dtype=np.float)

    # ret_val = [[tr_cod, tr_best, best_codtr, best_w_idx_tr, tr_avg_cod],
    #          [val_cod, val_best, best_codval, best_w_idx_val, val_avg_cod],
    #           [tr_avg_cod_avg_b, val_avg_cod_avg_b]]

    ret_val = [[lse_tr, lses_best_tr, t_b_lse_t, ret_tuple_trn[0][0], tr_avg_lse],
               [lse_val, lses_best_val, t_b_lse_v, ret_tuple_val[0][0], val_avg_lse],
               [tr_avg_lse_avg_b, val_avg_lse_avg_b]]

    return ret_val


# will attempt to train the data using the objects in
# a list of parameter values(w), training data, y for training data
# validation data, y for the validation data
def train_model_mse2(param_tr_val_a):
    w_list = param_tr_val_a[0]

    tr_l = param_tr_val_a[1]
    y_tr_l = param_tr_val_a[2]

    val_l = param_tr_val_a[3]
    y_val_l = param_tr_val_a[4]

    # ret_tuple,    w_tuple,     w_ret_l, sorted cods,  all the best cods, the best cod
    ret_tuple_trn, w_tuple_trn, tr_w_l, mse_tr, mses_best_tr, t_b_mse_t = test_data_set_mse_a(w_list, tr_l, y_tr_l)
    ret_tuple_val, w_tuple_val, val_w_l, mse_val, mses_best_val, t_b_mse_v = test_data_set_mse_a(tr_w_l, val_l, y_val_l)

    # tr_cod, tr_best, best_codtr, best_w_idx_tr = test_data_set_cod(w_list, tr_l, y_tr_l)
    # val_cod, val_best, best_codval, best_w_idx_val = test_data_set_cod(w_list, val_l, y_val_l)

    # ret_tuple_trn = test_data_set_cod_b(w_list, tr_l, y_tr_l)
    # ret_tuple_val = test_data_set_cod_b(w_list, val_l, y_val_l)

    print(format('\n'))
    print("Training set")
    print('ret tuple')
    print(ret_tuple_trn)
    print(ret_tuple_trn[0][0])
    print(ret_tuple_trn[0][1])
    print(w_tuple_trn)
    print(w_tuple_trn[0][0])
    print(w_tuple_trn[0][1])
    print(format('\n'))

    print("Validation set")
    print('ret tuple')
    print(ret_tuple_val)
    print(ret_tuple_val[0][0])
    print(ret_tuple_val[0][1])
    print(w_tuple_val)
    print(w_tuple_val[0][0])
    print(w_tuple_val[0][1])
    print(format('\n'))

    # grab the largest cod for each training and validation sets
    # tr_b_w = w_list[best_w_idx_tr[-1]]
    # val_b_w = w_list[best_w_idx_val[-1]]

    tr_b_w = w_list[int(ret_tuple_trn[0][0])]
    val_b_w = w_list[int(ret_tuple_val[0][0])]

    val_avg_mse_avg_b, tr_avg_mse_avg_b = get_avg_best_w_mse(w_list, tr_b_w, val_b_w, tr_l, y_tr_l, val_l, y_val_l)
    tr_avg_mse = np.mean(np.array(mse_tr, dtype=np.float), dtype=np.float)
    val_avg_mse = np.mean(np.array(mse_val, dtype=np.float), dtype=np.float)

    # ret_val = [[tr_cod, tr_best, best_codtr, best_w_idx_tr, tr_avg_cod],
    #          [val_cod, val_best, best_codval, best_w_idx_val, val_avg_cod],
    #           [tr_avg_cod_avg_b, val_avg_cod_avg_b]]

    ret_val = [[mse_tr, mses_best_tr, t_b_mse_t, ret_tuple_trn[0][0], tr_avg_mse],
               [mse_val, mses_best_val, t_b_mse_v, ret_tuple_val[0][0], val_avg_mse],
               [tr_avg_mse_avg_b, val_avg_mse_avg_b]]
    return ret_val

# ----------------------------------------------------------------------------------------------------------------
# ---------------------------------------------Trainers---------------------------------------------------------

# def train_cod_dos(x_d, y_d,)


# used to attain the best test size with the highest coefficient of determination
# can use split_array to either run multiple runs of the same size array to get an average error
# or have differnt sizes in the array to look for the best size vs cod
def train_model_cod_dos(x_data, y_data, split_array):
    # training_data, validation_data, y_training, y_validation
    print(format("\n"))
    print(len(x_data))
    # split_array = int(len(x_data) / 16)
    best_cod = 0
    cod_list = list()
    length_array = len(split_array)
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
            best_rand[0] = rand

    avg_cod = np.mean(np.array(cod_list, dtype=np.float), dtype=np.float)

    result_array = list([best_cod, cod_list, avg_cod])

    return result_array


# used to attain the best test size with the hightest coefficient of determination
def train_model_lse_dos(x_data, y_data, split_array):
    # training_data, validation_data, y_training, y_validation
    # print(len(Xdata))
    print(format("\n"))
    bs = list()
    best_ls = 1000
    ls_list = list()
    rand_list = list()
    best_rand = list(list([0]))
    # train_set, validation_set, y_training, y_validation = DataProcessor.dos_data_splitter(Xdata, Ydata, split_array)
    # while len(validation_set) >= 10:
    for x in range(0, len(split_array)):

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
    bs = list()
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


#                  -------------------------------------------------------
#                  -------------------------------------------------------
#                  -------------------------------------------------------
#                  -------------------------------------------------------
#                  -------------------------------------------------------
#                  -------------------------------------------------------
# used to attain the best test size with the highest coefficient of determination
# can use split_array to either run multiple runs of the same size array to get an average error
# or have differnt sizes in the array to look for the best size vs cod
def train_model_cod_tres(x_data, y_data, split_array):
    # training_data, validation_data, y_training, y_validation
    # print(format("\n"))
    # print(len(x_data))
    # split_array = int(len(x_data) / 16)

    best_cod = 0
    best_cod2 = 0
    cod_list = list()
    cod_list2 = list()
    length_array = len(split_array)
    best_rand = list(list([1]))
    rand_list = list()
    w_list = list()

    y_train_list = list()
    cod_tr_l = list()

    val_list = list()
    y_val_list = list()
    val_best_w = list()

    test_list = list()
    y_test_list = list()

    best_params = list()
    best_params2 = list()

    # while len(validation_set) >= 10:
    for x in range(0, length_array):
        tr, val, y_tr, y_val, ts, y_ts, rand = DataProcessor.tres_data_splitter(x_data, y_data, split_array[x])

        y_train_list.append(y_tr)

        w_params = multi_linear_regressor(tr, y_tr)

        train_g_model = get_r_data(tr, w_params)

        cod_tr_l.append(calculate_cod(train_g_model, y_tr))

        val_list.append(val)
        y_val_list.append(y_val)

        test_list.append(ts)
        y_test_list.append(y_ts)

        w_list.append(w_params)

        # use w to get some response data
        gmodel = get_r_data(val, w_params)
        cod = calculate_cod(gmodel, y_val)
        cod_list.append(cod)
        rand_list.append(rand)

        if best_cod < cod and cod > 0 and cod <= 1:
            # w_list.append(w_params)
            best_cod = cod
            best_params.append(len(w_list)-1)
            val_best_w.append(w_params)
            best_rand[0] = rand

    avg_cod1 = np.mean(np.array(cod_list, dtype=np.float), dtype=np.float)

    print(format('\n'))
    print("the best params are")
    print(best_params)
    print(format('\n'))

    # get test the w's on the test data
    for i in range(0, len(test_list)):

        for x in range(len(best_params)):
            g_model = get_r_data(test_list[i], w_list[x])
            cod = calculate_cod(g_model, y_ts)

            # if cod < best_cod2:
            if best_cod2 < cod and cod > 0 and cod <= 1:

                best_cod2 = cod
                best_params2.append(w_list[x])

    # average the parmeters so I can use the average of each one
    # as one set of parameters
    w_col_sum = list(w_list[0])
    # list((map(operator.add, listsum, w_list[l])))
    div_list = [(len(w_list))]

    for l in range(1, len(w_list)):
        w_col_sum = list(map(operator.add, w_col_sum, w_list[l]))
        div_list.append(len(w_list))

    avg_w = list(map(operator.truediv, w_col_sum, div_list))

    best_cod_avg = 1000

    cod_list_avg = list()

    print(format('\n'))
    print("the best params 2 are")
    print(best_params2)
    print(format('\n'))


    #  try using the averages of the best scoring parameters
    for i in range(0, len(test_list)):
        g_model = get_r_data(test_list[i], avg_w)
        cod = calculate_cod(g_model, y_ts)
        cod_list_avg.append(cod)
        if best_cod2 < cod and cod > 0 and cod <= 1:
            best_cod_avg = cod

    avg_cod_avg = np.mean(np.array(cod_list_avg, dtype=np.float), dtype=np.float)

    avg_cod2 = np.mean(np.array(cod_list2, dtype=np.float), dtype=np.float)

    ret_list = [[cod_list, best_cod, avg_cod1, rand_list, best_rand],
                [cod_list2, best_cod2, avg_cod2],
                [cod_list_avg, best_cod_avg, avg_cod_avg]]

    return ret_list


# used to attain the best test size with the highest coefficient of determination
# can use split_array to either run multiple runs of the same size array to get an average error
# or have differnt sizes in the array to look for the best size vs cod
def train_model_cod_tresB(x_data, y_data, split_array):
    # training_data, validation_data, y_training, y_validation
    # print(format("\n"))
    # print(len(x_data))
    # split_array = int(len(x_data) / 16)

    best_cod = 0
    best_cod2 = 0
    best_cod3 = 0
    bs = list()
    cod_list = list()
    cod_list2 = list()
    length_array = len(split_array)
    best_split = []
    best_rand = list(list([1]))
    rand_list = list()
    w_list = list()

    train_list = list()
    y_train_list = list()
    best_train_params = list()
    cod_tr_l = list()
    train_cod_avg = 0

    val_list = list()
    y_val_list = list()
    val_best_w = list()
    cod_val_l = list()
    val_cod_avg = 0


    test_list = list()
    y_test_list = list()
    test_best_w = list()
    cod_test_l = list()
    test_cod_avg = 0

    best_params = list()
    best_params2 = list()

    # while len(validation_set) >= 10:
    for x in range(0, length_array):
        tr, val, y_tr, y_val, ts, y_ts, rand = DataProcessor.tres_data_splitter(x_data, y_data, split_array[x])
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

        train_list.append(tr)
        y_train_list.append(y_tr)

        w_params = multi_linear_regressor(tr, y_tr)
        w_list.append(w_params)

        train_g_model = get_r_data(tr, w_params)

        cod_tr_l.append(calculate_cod(train_g_model, y_tr))

        val_list.append(val)
        y_val_list.append(y_val)

        test_list.append(ts)
        y_test_list.append(y_ts)

        if best_cod < cod_tr_l[-1] and cod_tr_l[-1] > 0 and cod_tr_l[-1] <= 1:
            best_cod = cod_tr_l[-1]
            #best_train_params.append(w_params)
            best_train_params.append(x)


    train_cod_avg = np.mean(np.array(cod_tr_l, dtype=float), dtype=float)

    '''
        # use w to get some response data
        gmodel = get_r_data(val, w_params)
        cod = calculate_cod(gmodel, y_val)
        cod_list.append(cod)
        rand_list.append(rand)

        if best_cod < cod and cod > 0 and cod <= 1:
            #w_list.append(w_params)
            best_cod = cod
            best_params.append(len(w_list)-1)
            val_best_w.append(w_params)
            bs = split_array[x]
            best_rand[0] = rand

    avg_cod1 = np.mean(np.array(cod_list, dtype=np.float), dtype=np.float)

    print(format('\n'))
    print("the best params are")
    print(best_params)
    print(format('\n'))
'''


    for i in range(len(val_list)):
        val_g = get_r_data(val_list[i], w_list[i])
        cod_val_l.append(calculate_cod(val_g, y_val_list[i]))
        if best_cod2 < cod_val_l[-1] and cod_val_l[-1] > 0 and cod_val_l[-1] <= 1:
            best_cod2 = cod_val_l[-1]
            #val_best_w.append(w_list[i])
            val_best_w.append(i)

    val_cod_avg = np.mean(np.array(cod_val_l, dtype=float), dtype=float)



    # get test the w's on the test data
    for i in range(0, len(test_list)):
        g_model = get_r_data(test_list[i], w_list[i])
        cod_test_l.append(calculate_cod(g_model, y_test_list[i]))
        if best_cod3 < cod_test_l[-1] and cod_test_l[-1] > 0 and cod_test_l[-1] <= 1:
            best_cod3 = cod_test_l[-1]
            #test_best_w.append(w_list[i])
            test_best_w.append(i)

    test_cod_avg = np.mean(np.array(cod_test_l, dtype=float), dtype=float)
    '''
    # average the parmeters so I can use the average of each one
    # as one set of parameters
    w_col_sum = list(w_list[0])
    # list((map(operator.add, listsum, w_list[l])))
    
    w_len = len(w_list)
    
    div_list = list(w_len)

    for l in range(1, len(w_list)):
        w_col_sum = list(map(operator.add, w_col_sum, w_list[l]))
        div_list.append(len(w_list))

    avg_w = list(map(operator.truediv, w_col_sum, div_list))

    best_cod_avg = 1000

    cod_list_avg = list()

    print(format('\n'))
    print("the best params 2 are")
    print(best_params2)
    print(format('\n'))


    #  try using the averages of the best scoring parameters
    for i in range(0, len(test_list)):
        g_model = get_r_data(test_list[i], avg_w)
        cod = calculate_cod(g_model, y_ts)
        cod_list_avg.append(cod)
        if best_cod2 < cod and cod > 0 and cod <= 1:
        #if cod < best_cod_avg:
            best_cod_avg = cod
'''


    # avg_cod_avg = np.mean(np.array(cod_list_avg, dtype=np.float), dtype=np.float)

    # avg_cod2 = np.mean(np.array(cod_list2, dtype=np.float), dtype=np.float)

    ret_list = [[cod_tr_l, best_cod, best_train_params, train_cod_avg],
                [cod_val_l, best_cod2, val_best_w, val_cod_avg],
                [cod_test_l, best_cod3, test_best_w, test_cod_avg]]

    return ret_list


# used to attain the best test size with the hightest coefficient of determination
def train_model_lse_tres(x_data, y_data, split_array):
    # training_data, validation_data, y_training, y_validation
    # print(len(Xdata))

    print(format("\n"))

    bs = list()
    best_ls = 1000
    best_mse = 1000
    best_mse2 = 1000
    ls_list = list()
    ls_list2 = list()
    ls_list_avg = list()
    rand_list = list()
    best_rand = list(list([0]))
    w_list  = list()
    test_list = list()
    y_test_list = list()

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

        tr, val, y_tr, y_val, ts, y_ts, rand = DataProcessor.tres_data_splitter(x_data, y_data, split_array[x])

        # get the parameters from the regression
        w_params = multi_linear_regressor(tr, y_tr)

        test_list.append(ts)
        y_test_list.append(y_ts)
        w_list.append(w_params)

        # use w to get some response data
        gmodel = get_r_data(val, w_params)
        ls = least_squares_estimate(gmodel, y_val)
        ls_list.append(ls)
        rand_list.append(rand)
        if ls < best_ls:
            best_ls = ls
            bs = split_array[x]
            best_rand[0] = rand

    avg_ls1 = np.mean(np.array(ls_list, dtype=np.float), dtype=np.float)

    # get test the w's on the test data
    for i in range(0, len(w_list)):

        g_model = get_r_data(test_list[i], w_list[i])
        lse = least_squares_estimate(g_model, y_ts)
        ls_list2.append(lse)

        if lse < best_mse2:
            best_lse2 = lse

    # average the parmeters so I can use the average of each one
    # as one set of parameters
    w_col_sum = list(w_list[0])
    # list((map(operator.add, listsum, w_list[l])))
    div_list = [(len(w_list))]

    for l in range(1, len(w_list)):
        w_col_sum = list(map(operator.add, w_col_sum, w_list[l]))
        div_list.append(len(w_list))

    avg_w = list(map(operator.truediv, w_col_sum, div_list))

    best_ls_avg = 1000

    ls_list_avg = list()

    for i in range(0, len(test_list)):
        g_model = get_r_data(test_list[i], avg_w)
        lse = least_squares_estimate(g_model, y_ts)
        ls_list_avg.append(lse)
        if lse < best_ls_avg:
            best_lse_avg = lse

    avg_lse_avg = np.mean(np.array(ls_list_avg, dtype=np.float), dtype=np.float)

    avg_lse2 = np.mean(np.array(ls_list2, dtype=np.float), dtype=np.float)

    ret_list = [[ls_list, best_ls, avg_ls1, rand_list, best_rand],
                [ls_list2, best_lse2, avg_lse2],
                [ls_list_avg, best_lse_avg, avg_lse_avg]]

    return ret_list


# used to attain the best test size with the highest coefficient of determination
def train_model_mse_tres(x_data, y_data, split_array):
    # training_data, validation_data, y_training, y_validation
    print(format("\n"))
    # bs = list()
    best_mse = 1000
    best_mse2 = 1000
    mse_list = list()
    mse_list2 = list()
    best_rand = list(list([0]))
    rand_list = list()
    w_list = list()
    test_list = list()
    y_test_list = list()

    for x in range(0, len(split_array)):
        tr, val, y_tr, y_val, ts, y_ts, rand = DataProcessor.tres_data_splitter(x_data, y_data, split_array[x])

        w_params = multi_linear_regressor(tr, y_tr)

        test_list.append(ts)
        y_test_list.append(y_ts)
        w_list.append(w_params)

        # use w to get some response data
        g_model = get_r_data(val, w_params)
        rand_list.append(rand)
        mse = mean_square_error(g_model, y_val)
        mse_list.append(mse)
        rand_list.append(rand)

        if mse < best_mse:
            best_mse = mse
            # bs = split_array[x]
            best_rand[0] = rand

    avg_mse1 = np.mean(np.array(mse_list, dtype=np.float), dtype=np.float)

    for i in range(0, len(w_list)):
        g_model = get_r_data(test_list[i], w_list[i])
        mse = mean_square_error(g_model, y_ts)
        mse_list2.append(mse)

        if mse < best_mse2:
            best_mse2 = mse

    w_col_sum = list(w_list[0])
    # list((map(operator.add, listsum, w_list[l])))
    div_list = [(len(w_list))]

    for l in range(1, len(w_list)):
        w_col_sum = list(map(operator.add, w_col_sum, w_list[l]))
        div_list.append(len(w_list))

    avg_w = list(map(operator.truediv, w_col_sum, div_list))

    best_mse_avg = 1000

    mse_list_avg = list()

    for i in range(0, len(test_list)):
        g_model = get_r_data(test_list[i], avg_w)
        mse = mean_square_error(g_model, y_ts)
        mse_list_avg.append(mse)
        if mse < best_mse_avg:
            best_mse_avg = mse

    avg_mse_avg = np.mean(np.array(mse_list_avg, dtype=np.float), dtype=np.float)

    avg_mse2 = np.mean(np.array(mse_list2, dtype=np.float), dtype=np.float)

    ret_list = [[mse_list, best_mse, avg_mse1, rand_list, best_rand],
                [mse_list2, best_mse2, avg_mse2],
                [mse_list_avg, best_mse_avg, avg_mse_avg]]

    return ret_list

# -------------------------------------------------------------------------------------------------------------

# ------------------------------------------------Error Functions----------------------------------------------


# calculates the Mean Square Error
def calculate_cod(g_model, r_validate):

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
        difference_list.append(pow(diff, 2))
        # difference_list.append(np.absolute(diff))
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
# get a set of y's using x_data and parameters w
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
        # x_observation.append(1)
        for col in range(len(x_data[0])):
            x_observation.append(x_data[row][col])
        r.append(np.dot(np.array(x_observation, dtype=np.float64), wnp))
    return r


# used for linear regression imputation
def getlinregmissingdata(regdata, baddic, w):
    r = []

    # print('----------------------------------------------------------------regdata')
    # print(regdata)

    for entry in baddic:
        dlist = baddic[entry]
        for row in dlist:
            # print('------------row')
            # print(row)
            x = list()
            x.append(1)
            for col in range(len(regdata[0])-1):
                if col != entry:
                    x.append(regdata[row][col])
            Xnp = np.array(x, dtype=np.float64)
            Wnp = np.array(w, dtype=np.float64)

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

    cols_f = list()

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
                f_tmp = F[:]
                # Fsaver.count()

                # create a temp F array
                # each row of Ftmp contains a list
                # adds the current column
                for row in range(nx):
                    f_tmp[row].append(x_data[row][col])

                #       print('in for Ftmp')
                #      print(Ftmp)

                # split the data into training and validation sets
                train, validation, y_training, y_validation, rand = DataProcessor.dos_data_splitter(f_tmp, y_data, split)

                # perform linear regression to get W params
                w_params = multi_linear_regressor(train, y_training)

                # use w to get some response data
                gmodel = get_r_data(validation, w_params)

                # calculate the mean square error for this x column
                mse = mean_square_error(gmodel, y_validation)
                for row in range(nx):
                    f_tmp[row].pop()
                #     print('Ftmp is now')
                #    print(Ftmp)
                #   print(format("\n"))
                #   print('new mse')
                #   print(mse)

                if mse < mininmum_mse[0]:
                    #      print('found new min mse as '+ str(mse) + ' at col ' + str(col))
                    mininmum_mse[0] = mse
                    addcol[0] = col
                    cols_f.append(col)
                    found = True
                    # Fsaver = list(Ftmp)
                del f_tmp[col]
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


# attempts to do forward selection
def forward_selector_test(x_data, y_data, split):

    nx = len(x_data)
    # print('X data is ')
    # print(x_data)
    # print(len(x_data))
    ny = len(y_data)
    col_size = len(x_data[0])
    used_col = []

    cols_f = list()

    found = True

    addcol = [2000]

    mininmum_mse = [10000]

    F = list()
    Fsaver = list()

    # find the first variable  array to add to F as well its mean square error
    min_col, min_mse, best_col = find_first(x_data.copy(), y_data.copy(), split)

    '''
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
    '''

    cols_f.append(min_col[0])

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
                f_tmp = F[:]
                # Fsaver.count()

                # create a temp F array
                # each row of Ftmp contains a list
                # adds the current column
                for row in range(nx):
                    f_tmp[row].append(x_data[row][col])

                #       print('in for Ftmp')
                #      print(Ftmp)

                # split the data into training and validation sets
                train, validation, y_training, y_validation, rand = DataProcessor.dos_data_splitter(f_tmp, y_data, split)

                # perform linear regression to get W params
                w_params = multi_linear_regressor(train, y_training)

                # use w to get some response data
                gmodel = get_r_data(validation, w_params)

                # calculate the mean square error for this x column
                mse = mean_square_error(gmodel, y_validation)
                for row in range(nx):
                    f_tmp[row].pop()
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
                del f_tmp[col]
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
            cols_f.append(addcol[0])
            if len(F) == col_size + 1:
                break
        # add best column
        # for row in range(nx):
        # flist.append(x_add[row])

        # for row in range(nx):
        #    F[row].append(flist)

    return F, mininmum_mse[0], cols_f


# ------------------------------------------Display functions--------------------------------------------------
# will display the original data array and the dependent and independent portions
def show_data_x_y(d_a, x, y):

    print('Data Array:')
    print(d_a)
    print('Dependent Data:')
    print(x)
    print('Independent Data:')
    print(y)
    return


def show_stat_array(stat_a):
    print(format('\n'))
    print('sample mean array')
    print(stat_a[0])
    print(format('\n'))
    print('sample std array')
    print(stat_a[1])
    print(format('\n'))
    print('Min array')
    print(stat_a[2])
    print(format('\n'))
    print('Max array')
    print(stat_a[3])
    print(format('\n'))
    return


def show_test_results(imp_name, err_name, result_l, prec1, prec2):
    print('Imputation type: ' + imp_name)
    print('Error Checking  method: ' + err_name)

    # tr1_info = ret_list[0]
    # val1_info = ret_list[1]
    # avg_cod_avg_val = ret_list[2][1]
    # avg_cod_avg_tr = ret_list[2][0]

    # val2_info = ret_list[1]

    # [tr_cod, tr_best, best_codtr, best_w_idx_tr, tr_avg_cod ],

    print('-----training info------')
    print('Best Training ' + err_name)
    print(result_l[0][1])
    print('Best '+ err_name + ' cod idx')
    print(result_l[0][3])
    print('best training avg ' + err_name)
    print(np.around(result_l[0][4], prec1))

    print('-----validation info------')
    print('Best Validation ' + err_name)
    print(result_l[1][1])
    print('Best Validation ' + err_name + ' idx')
    print(result_l[1][3])
    print('Best Validation avg ' + err_name)
    print(np.around(result_l[1][4], prec1))

    print('------avg using an avg w---')
    print('Training new avg ' + err_name)
    print(np.around(result_l[2][0], prec2))
    print('Validation new avg ' + err_name)
    print(np.around(result_l[2][1], prec2))

    return

# ----------------------------------------------------------------------------------------------------------

