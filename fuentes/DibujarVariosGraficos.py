# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:40:48 2017

@author: maryluz
"""
from os import listdir
from os.path import isfile, join
import os
import os.path as path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

def Menu ():
    
    while True:
        os.system('cls')
        print("Choose an option:")
        print("1. Betweeness graphics")
        print("2. Pagerank graphics")
        print("3. Degree graphics")
        print("4. Eigenvector graphics")
        print("5. Random node graphics")
        print("6. Clustering graphics")

        print("7. Exit")
       
        option = input("")
        
        if (option =='1') or (option =='2') or (option =='3') or (option == '4') or (option == '5') or (option =='6') or (option =='7'):
            break
        
    return option
        
def DibujarGraficos (Grafico, pC, pR):
	mi_path = os.getcwd() + "\\" + "..\\results\\Propagacion"+pC+"_"+pR+"\\Graficos\\"+Grafico
	fig = plt.figure()
	ax = plt.subplot(111)
	plt.title('Failure propagation')
	ax.set_xlabel('n')
	ax.set_ylabel('Infection rate')
	fmt = '%.2f' # Format you want the ticks, e.g. '40%'
	xticks = mtick.FormatStrFormatter(fmt)
	ax.yaxis.set_major_formatter(xticks)

	datosHeader = ['iter', 'rate']

	for NomFich in listdir(mi_path):
		print (NomFich)
		pathFile = os.getcwd() + "\\" + "..\\results\\Propagacion"+ pC + "_" +pR+ "\\Graficos\\"+ Grafico +"\\" + NomFich
		datos = pd.read_table(pathFile, 'engine=python', delimiter=' ', header=0, encoding = "ISO-8859-1", names=datosHeader)
		datos=datos[(datos.iter!=0)&(datos.rate!=0)]
		ax.plot(datos['iter'], datos['rate'], label='$y = a')
	fileFig="TodosGraficos"+Grafico
	print("Nombre de fichero")
	print(fileFig)
	pathFileFig = os.getcwd() + "\\" + "..\\results\\Propagacion"+pC+"_"+pR+"\\Graficos\\"+Grafico+"\\"+fileFig
	print("El path es")
	print(pathFileFig)
	fig.savefig(pathFileFig)
       

option = Menu()  
option = int(option)
print ("Probabilidad de contagio")
pC = input("")
print ("Probabilidad de recuperacion")
pR = input("")
if (option == 1):
   DibujarGraficos('B', str(pC), str(pR))
elif (option== 2):
   DibujarGraficos('P', pC, pR)
elif (option == 3): 
   DibujarGraficos('D', pC, pR)
elif (option == 4):
   DibujarGraficos('E', pC, pR)
elif (option == '5'):
   DibujarGraficos('R', pC, pR)
else:
   DibujarGraficos('C', pC, pR)
