import numpy as np
import pandas as pd
import matplotlib
import os
import os.path as path



boxprops = dict(linestyle='--', linewidth=3, color='darkgoldenrod')
flierprops = dict(marker='o', markerfacecolor='green', markersize=8,linestyle='none')
medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')


fileName = '..\datos\TodosD.txt'
fileFig = '..\datos\FigTodosD.png'

pathFile = os.getcwd() + "\\" + fileName
headerD = ['car','time','rate']
todosD = pd.read_table(pathFile, 'engine=python', delimiter=' ', header=0, encoding = "ISO-8859-1", names=headerD)

myPlot= todosD.boxplot(column="rate", by= "car", patch_artist=False, flierprops=flierprops, boxprops=boxprops, showmeans=False, meanline=False, medianprops=medianprops, figsize= (8,8)) 



myFig = myPlot.get_figure()  
pathFile = os.getcwd() + "\\" +  fileFig         # Get the figure
myFig.savefig(pathFile)  # Save to file
