import DataCleaner as DC
import GDataWorks as GDW
import numpy as np

#get the data using data cleaner
dataarray = DC.DataCleaner("CarData.txt")

#Find the bad data and store it in a map keyed on the column
#where the bad data was found and with the rows in that column
#where
baddatdic = GDW.FindColBadData(dataarray)

adjusteddata = GDW.ReplaceBadData(dataarray, baddatdic, str(0))
adjusteddatafloat = GDW.MakeDataFloats(adjusteddata, 8)

print(adjusteddatafloat)
