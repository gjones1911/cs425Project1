'''
Created by: Gerald Jones
Purpose: This will hopefully use linear regression to model how
         the number of cylinders, displacement, horsepower, weight,
         acceleration, model year, and origin of a car can be used with
         linear regression techniques to predict mpg. This one discards
         observations with bad data
'''


import numpy as np
from matplotlib.pyplot import *
import GDataWorks as GDW
import DataCleaner as DC

error_array = ['COD', 'Least_squares', 'MSE']

# get the data using data cleaner
dataarray = DC.DataCleaner("CarData.txt")


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
adjusteddata = GDW.discardbaddata(dataarray, baddatdic)
print('adjustedata')
print(adjusteddata[32])
print(dataarray[32])

origdata = list(dataarray)



# adjust the data by making part of it numerical(floats) and giving it a column to
# stop the operation
adjusteddatafloat = GDW.MakeDataFloats(list(adjusteddata), 8)


# get all the mpg data as a list
mpgvals, baddatampg = DC.GetCol(adjusteddatafloat.copy(), 0)
npmpgvals = np.array([mpgvals])
npmpgvalsmean = np.mean(npmpgvals)

print('mpg mean')
print(npmpgvalsmean)

smu = GDW.samplemeanarray(adjusteddatafloat.copy())

print('smu')
print(smu)

W, Xx, Yy = GDW.MulitLinearRegressor(adjusteddatafloat, 0)

print('W')
print(len(W))
print(W)

print('adjusted data as float')
print(len(adjusteddatafloat))

print('X')
print(len(Xx))
print(len(Xx[0]))
print(Xx)


print('Y')
print(len(Yy))
print(1)
print(Yy)

# Coefficient of Determination :
best_cod, bs = GDW.TrainModel(Xx, Yy, error_array[0])
print('best_cod Cod')
print(best_cod)
print('best split Cod')
print(bs)

# Lest Squares Error :
best_ls, bs = GDW.TrainModel(Xx, Yy, error_array[1])
print('best Least Squares erro')
print(best_ls)
print('best split Least Squares error')
print(bs)

# Meas Square Error :
best_mse, bs = GDW.TrainModel(Xx, Yy, error_array[2])
print('best MSE')
print(best_mse)
print('best split MSE')
print(bs)
