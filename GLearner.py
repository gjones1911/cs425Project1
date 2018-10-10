import DataCleaner as DC
import GDataWorks as GDW
import numpy as np

#get the data using data cleaner
dataarray = DC.DataCleaner("CarData.txt")

#Find the bad data and store it in a map keyed on the column
#where the bad data was found and with the rows in that column
#where the bad data is as the vals as a list
baddatdic = GDW.FindColBadData(dataarray)

#TODO: create a function to use linear regresion to replace bad data



#adjust the data by replacing bad data with some value say a string version of 0
adjusteddata = GDW.ReplaceBadData(dataarray, baddatdic, str(0))

#adjust the data by making part of it numerical(floats) and giving it a column to
#stop the operation
adjusteddatafloat = GDW.MakeDataFloats(adjusteddata, 8)


#function to use averageing to replace bad data
avgdata = GDW.AverageReplacer(adjusteddatafloat, baddatdic, 0.0)




test = [1,2,3]
#print(sum(test))
print(avgdata)
