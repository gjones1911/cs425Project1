'''
Created by: Gerald Jones
Purpose: This will hopefully use linear regression to model how
         the number of cylinders, displacement, horsepower, weight,
         acceleration, model year, and origin of a car can be used with
         linear regression techniques to predict mpg. This one uses the average
         of an attrribute set with bad data to replace missing data
'''

import numpy as np
from matplotlib.pyplot import *
import GDataWorks as GDW
import DataCleaner as DC

# get the data using data cleaner
dataarray = DC.DataCleaner("CarData.txt")

error_array = ['COD', 'Least_squares', 'MSE']

#print('data0')
#print(dataarray[32])
# Find the bad data and store it in a map keyed on the column
# where the bad data was found and with the rows in that column
# where the bad data is as the vals as a list
baddatdic = GDW.FindColBadData(dataarray.copy(), '?')

print(baddatdic)

#print('data1')
#print(dataarray[32])
# adjust the data by replacing bad data with some value say a string version of 0
adjusteddata = list(GDW.ReplaceBadData(list(dataarray), baddatdic, str(0)))
#print('data2')
#print(dataarray[32])

origdata = list(dataarray)
# adjust the data by making part of it numerical(floats) and giving it a column to
# stop the operation
adjusteddatafloat = GDW.MakeDataFloats(list(adjusteddata), 8)

'''
print('original data')
print(origdata[32])
print('adjusted data')
print(adjusteddata[32])
print('adjusteddatafloat')
print(adjusteddatafloat[32])
'''
# function to use averaging to replace bad data
avgdata = list(GDW.AverageReplacer(adjusteddatafloat.copy(), baddatdic, 0.0))

#print('avgdata[32]')
#print(avgdata[32])

# get all the mpg data as a list
mpgvals, baddatampg = DC.GetCol(avgdata.copy(), 0)
npmpgvals = np.array([mpgvals])
npmpgvalsmean = np.mean(npmpgvals)

#print('mpg mean')
#print(npmpgvalsmean)

smu = GDW.samplemeanarray(avgdata.copy())
print('smu')
print(smu)

sstd2 = GDW.sample_std_array(avgdata.copy())
print('sample std ')
print(sstd2)

W, Xx, Yy = GDW.MulitLinearRegressor(avgdata, 0)

print('W')
print(len(W))
print(W)

print('X')
print(len(Xx))
print(len(Xx[0]))
print(Xx)

print('Y')
print(len(Yy))
print(1)
print(Yy)

# Coefficient of Determination :
best_cod, bs = GDW.TrainModelCOD(Xx, Yy)
print('best_cod Cod')
print(best_cod)
print('best split Cod')
print(bs)
print(format("\n"))

# Lest Squares Error :
best_ls, bs = GDW.TrainModelLSE(Xx, Yy)
print('best Least Squares error')
print(best_ls)
print('best split Least Squares error')
print(bs)
print(format("\n"))

# Meas Square Error :
best_mse, bs = GDW.TrainModelMSE(Xx, Yy)
print('best MSE')
print(best_mse)
print('best split MSE')
print(bs)
print(format("\n"))
split = bs

#F = GDW.forward_selector(list(Xx), list(Yy), int(398/2))

#x_col = GDW.GetColumn(avgdata, 1)

#print('x_col')
#print(len(x_col))
#print(x_col)
'''
min_col, min_mse, best_col = GDW.find_first(Xx, Yy)


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

#m_1, b_1, X_1, Y_1, Yg_1 = GDW.reg_lin_regression(list(x_col), list(Yy))

print('m_1')
print(m_1)

print('b_1')
print(b_1)
'''

F, best_mse = GDW.forward_selector(Xx, Yy, split)

print('best MSR is ')
print(best_mse)
print('F')
print(F)

