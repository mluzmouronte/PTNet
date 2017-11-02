import numpy as np
import pandas as pd
import matplotlib
import os
import os.path as path
from scipy import stats  
from scipy import stats
fileName = '..\datos\TodosD.txt'

pathFile = os.getcwd() + "\\" + fileName
headerD = ['car','time','rate']
todosD = pd.read_table(pathFile, 'engine=python', delimiter=' ', header=0, encoding = "ISO-8859-1", names=headerD)

datos = todosD[todosD.car == 'B']
datos = list(datos["rate"].dropna())  
print(datos)
f_val, p_val=stats.f_oneway(datos,datos)  
print(type(p_val))
print ("One-way ANOVA P =")
print(p_val)  

