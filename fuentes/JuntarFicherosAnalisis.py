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


def UnirFicheros (pC, pR):
	mi_path = os.getcwd() + "\\" + "..\\results-ufv\\Propagacion"+pC+"_"+pR+"\\"
	masterDF = pd.DataFrame([['',0,0]], columns = ['ch','iter', 'rate'])
	datosHeader = ['iter', 'rate']
	for NomFich in listdir(mi_path):
		print (NomFich)
		pathFile = os.getcwd() + "\\" + "..\\results-ufv\\Propagacion"+ pC + "_" +pR+ "\\"+ NomFich
		datos = pd.read_table(pathFile, 'engine=python', delimiter=' ', header=0, encoding = "ISO-8859-1", names=datosHeader)
		datos=datos[(datos.iter!=0)&(datos.rate!=0)]
		datosAux=datos
		if (NomFich.find("_B-") >=0):
			datosAux['ch'] = 'B'
		elif (NomFich.find("_A-") >=0):
			datosAux['ch'] = 'A'
		elif (NomFich.find("_C-") >=0):
			datosAux['ch'] = 'C'
		elif (NomFich.find("_D-") >=0):
			datosAux['ch'] = 'D'
		elif (NomFich.find("_E-") >=0):
			datosAux['ch'] = 'E'
		elif (NomFich.find("_P-") >=0):
			datosAux['ch'] = 'P'
		else:
			datosAux['ch'] = 'N'
		masterDF = masterDF.append(datosAux)
	#print("Master DF")
	#print(masterDF)
	masterDF=masterDF[masterDF.ch !='']
	pathFile = os.getcwd() + "\\" + "..\\results-ufv\\"+"TodosDate"+pC+pR
	masterDF=masterDF[masterDF.ch!=""]
	masterDF.to_csv(pathFile, header=None, index=None, mode='a', sep=' ')

print ("Probabilidad de contagio")
pC = input("")
print ("Probabilidad de recuperacion")
pR = input("")
UnirFicheros(pC, pR)
