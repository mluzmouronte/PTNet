# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:27:54 2017

@author: maryluz.mouronte
"""

"""
Editor de Spyder

Este es un archivo temporal
"""

import pandas as pd
import numpy as np
import os
import os.path as path
import numpy as np
import pandas as pd
import networkx as nx
import operator
import random
from random import randint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from os import listdir
import shutil
import time
from collections import OrderedDict
import copy


#Variables globales para optimizar

NodeInfo = pd.DataFrame([[0,0]], columns = ['node','status'], dtype=int)
V = nx.Graph()
Rate = 0
pC=1
pR=0
NDir=''
pathFile=''
Time = 1
TotalNodes = 0

#Fin variables globales para optimizar

NO_INFECTADO=0
INFECTADO=1
MAX_TIME=700
MAX_ITER=100


def LeerFicheros():
 
    stopsHeader = ['stop_id','stop_code','stop_name','stop_desc','stop_lat','stop_lon','location_type','parent_station']

    fileName = '..\datos\F_stops.txt'

    pathFile = os.getcwd() + "\\" + fileName

    
    #stops = pd.read_csv(pathFile, sep=';', header=0, encoding = "ISO-8859-1", names=stopsHeader)
    stops = pd.read_table(pathFile, 'engine=python', delimiter=',', header=0, encoding = "ISO-8859-1", names=stopsHeader)

    #Viajes
    #tripsHeader = ['route_id', 'service_id','trip_id','trip_headsign','trip_short_name','direction_id', 'block_id', 'shape_id']
   
    tripsHeader = ['route_id', 'service_id','trip_id','trip_headsign','trip_short_name','direction_id','shape_id']

    fileName = '..\datos\F_trips.txt'

    pathFile = os.getcwd() + "\\" + fileName

    
    #trips = pd.read_csv(pathFile, sep=';', header=0, encoding = "ISO-8859-1", names=tripsHeader)

    trips = pd.read_table(pathFile, 'engine=python', delimiter=',', header=0, encoding = "ISO-8859-1", names=tripsHeader)

    #print(pathFile)
    
    #print('Viajes')
    #print(stops.head())

    #stops_times


    stopstimesHeader =['trip_id','arrival_time','departure_time','stop_id','stop_sequence','stop_headsign','pickup_type','drop_off_type','shape_dist_traveled']
    
    #pickup_type,drop_off_type,shape_dist_traveled

    fileName = '..\datos\F_stop_times.txt'
    

    pathFile = os.getcwd() + "\\" + fileName
    #print(pathFile)
    stoptimes = pd.read_table(pathFile, 'engine=python', delimiter=',', header=0, encoding = "ISO-8859-1", names=stopstimesHeader)
    #trips = pd.read_csv(pathFile, sep=';', header=0, encoding = "ISO-8859-1", names=tripsHeader)
    #stoptimes=trips
    #pathFile = pathFile + 'sal'
    #stoptimes.to_csv(pathFile, sep=';')
    
    #print(stoptimes.head())
    
    #miDf = pd.merge(stoptimes, trips)
    return stoptimes, trips, stops

def ConstruirGrafo(fileName):
    
    print("Building graph")
    
    pathFile = os.getcwd() + "\\" + "..\\datos\\" + fileName
                    
    with open(pathFile) as f:
        lines = f.readlines()
        
    linesN=[]
    for line in lines:
       line= line.replace('\x00','')
       if (len(line) != 0):
           linesN.append(line)
    lines = linesN

    myList = [line.strip().split() for line in lines]
    # [['a', 'b'], ['a', 'c'], ['b', 'd'], ['c', 'e']]
    
    myListInt= [[int(y) for y in x] for x in myList]
    
    #print(myListInt)
    
    g = nx.Graph()
    #g.add_weighted_edges_from(myList)
    
    g.add_edges_from(myListInt)

   
    
    #nx.write_edgelist(CG,'saida.dat')
    
    return g

def ObtenerNodos (List):
    
    Nodes = []
        
    sorted_x = sorted(List.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    
    print("***Lista***")
    print(sorted_x)
    
    for i in range (len(sorted_x)):
      Tupla =sorted_x[i]
      Nodes.append(Tupla[0])
    
    print("***Nodos***")
    print(Nodes)
    
    return Nodes
    

   
def ObtenerPropiedadesGrafo (Grafo):
    
    #Calculamos el betweenness
    
   
    Betweenness = nx.betweenness_centrality(Grafo, normalized=True, weight=None)
    
    #Calculamos el grado
   
    
    Degree=Grafo.degree()
    
    #Calculamos el PageRank
    PageRank = nx.pagerank(Grafo, alpha=0.9)

    
    #Calculamos la centralidad
    Centrality = nx.eigenvector_centrality_numpy(Grafo)
    
    #Calculamos el clustering
    Clustering = nx.clustering(Grafo)
    
    
    
    Nodes=Grafo.nodes()
    Alea=[]
    for i in range (1, len(Nodes)):
        Node=randint(1, len(Nodes))
        Alea.append(Nodes[Node-1])
    Alea=list(set(Alea))
    
   
    
    return Betweenness, Degree, PageRank, Centrality, Alea, Clustering
    
def QuitarNodos (g, Nodes):
    
    g.remove_nodes_from(Nodes)
    
    return (g)

def CalcularCG (g, pathFile, NumNodesRem):
    
    CG=sorted(nx.connected_components(g), key = len, reverse=True)
    #print(g.edges())
    
    #fileName=fileName.replace(".txt", "GC.txt")
    #pathFile = os.getcwd() + "\\" + fileName
   
    
    GigantC= pd.DataFrame(index=[0], columns=('N','GC'))
    GigantC= GigantC.fillna(0) # with 0s rather than NaNs
    GigantC['N']=NumNodesRem
    GigantC['GC']=len(CG[0])
    
    #print(GigantC)
 
    GigantC.to_csv(pathFile, header=None, index=None, mode='a', sep=' ')
    GigantC = GigantC.drop(labels=[0], axis=0) #Probar
   
    
def CalcularAislamiento (G, MaxNodes, pathFile, List, ExtraerN):
    
    
    
    Grafo = copy.deepcopy(G)
    ElemBloq=5
       
    if ExtraerN:
        Nodos= ObtenerNodos(List)
    else:
        Nodos=List
    NumBloq= int(len(Nodos)/ElemBloq)
    
    
    CalcularCG (Grafo, pathFile, 0)
    for i in range(1, NumBloq+1): 
        Grafo = copy.deepcopy(G)
        N= Nodos[0:i*ElemBloq]
        Grafo= QuitarNodos(Grafo,N)
        CalcularCG (Grafo, pathFile, len(N) )

def CalcularRobustez (fileName):
    
    
    G = ConstruirGrafo(fileName)
    Grafo = copy.deepcopy(G)
    Bet, Deg, Pag, Eig, Alea, Clustering = ObtenerPropiedadesGrafo(Grafo)
    MaxNodes = len(Grafo.nodes())
    
    
    #Betweeness
    pref = "CGBet"
    pathFile = os.getcwd() + "\\" + "..\\datos\\" + pref + fileName
                        
    print(pathFile)
    if path.exists(pathFile):
        os.remove(pathFile)  
    CalcularAislamiento (Grafo, MaxNodes, pathFile, Bet, True)
    
    Grafo.clear()
    Grafo = ConstruirGrafo(fileName)
    #Degree
    pref = "CGDeg"
    pathFile = os.getcwd() + "\\" + "..\\datos\\" + pref + fileName
             
    print(pathFile)
    if path.exists(pathFile):
        os.remove(pathFile)
    CalcularAislamiento (Grafo, MaxNodes, pathFile, Deg, True)   
    
    
    Grafo.clear()
    Grafo = ConstruirGrafo(fileName)    
    #PageRank
    pref = "CGPag"
    pathFile = os.getcwd() + "\\" + "..\\datos\\" + pref + fileName
                        
    print(pathFile)
    if path.exists(pathFile):
       os.remove(pathFile)
    CalcularAislamiento (Grafo, MaxNodes, pathFile, Pag, True)       
    
    Grafo.clear()
    Grafo = ConstruirGrafo(fileName)    
    #Eigenvector
    pref = "CGEig"
    pathFile = os.getcwd() + "\\" + "..\\datos\\" + pref + fileName
                        
    print(pathFile)
    if path.exists(pathFile):
       os.remove(pathFile)
    CalcularAislamiento (Grafo, MaxNodes, pathFile, Eig, True)   

    Grafo.clear()
    Grafo = ConstruirGrafo(fileName)    
    #Aleatorio
    pref = "CGAlea"
    pathFile = os.getcwd() + "\\" + "..\\datos\\" + pref + fileName
                        
    print(pathFile)
    if path.exists(pathFile):
       os.remove(pathFile)
    CalcularAislamiento (Grafo, MaxNodes, pathFile, Alea, False)        
    

def ObtenerSpaceL():
    
    fileName = '..\datos\SpaceL.txt'

    pathFile = os.getcwd() + "\\" + fileName
    if path.exists(pathFile):
        os.remove(pathFile)
      
    infoStopTimes, tripsID, infoStops = LeerFicheros()
    
    
    
    infoStopTimes=infoStopTimes.drop(labels="arrival_time", axis=1)
    infoStopTimes=infoStopTimes.drop(labels="departure_time", axis=1)
    infoStopTimes=infoStopTimes.drop(labels="stop_headsign", axis=1)
    infoStopTimes=infoStopTimes.drop(labels="shape_dist_traveled", axis=1)
    infoStopTimes=infoStopTimes.drop(labels="pickup_type", axis=1)
    infoStopTimes=infoStopTimes.drop(labels="drop_off_type", axis=1)
    infoStopTimes = infoStopTimes.drop_duplicates()
    
    trips=infoStopTimes['trip_id'].drop_duplicates()
   
    
    for i in trips:
    #i=10000872820081804
        data_filter =  infoStopTimes[infoStopTimes['trip_id'] == i]
        nodes2 = data_filter["stop_id"]
        nodes2=nodes2.drop(nodes2.index[0])
       
        nodes1=data_filter["stop_id"]
        Count_Row=nodes1.shape[0] - 1 #gives number of row count
        nodes1=nodes1.drop(nodes1.index[Count_Row])
        
        nodes1 = pd.DataFrame(nodes1)
        dupla=nodes1.reset_index()
        dupla=dupla.drop(labels="index", axis=1)
        nodes2 = pd.DataFrame(nodes2)
        nodes2=nodes2.reset_index()
        nodes2=nodes2.drop(labels="index", axis=1)
    
   
        dupla.columns = ['node1']
        nodes2.columns = ['node2']
        dupla=dupla.join(nodes2)
        dupla = dupla.drop_duplicates()
        
        #dupla=dupla.groupby(['node1', 'node2']).size().reset_index().rename(columns={0:'count'}) con pesos
       
        dupla.to_csv(pathFile, header=None, index=None, mode='a', sep=' ')
   


def ObtenerSpaceP():
      
    fileName = '..\datos\SpaceP.txt'
    
    pathFile = os.getcwd() + "\\" + fileName
    if path.exists(pathFile):
        os.remove(pathFile)
    
    infoStopTimes, tripsID, infoStops = LeerFicheros()
    
    
    
    infoStopTimes=infoStopTimes.drop(labels="arrival_time", axis=1)
    infoStopTimes=infoStopTimes.drop(labels="departure_time", axis=1)
    infoStopTimes=infoStopTimes.drop(labels="stop_headsign", axis=1)
    infoStopTimes=infoStopTimes.drop(labels="shape_dist_traveled", axis=1)
    infoStopTimes = infoStopTimes.drop_duplicates()
    
    trips=infoStopTimes['trip_id'].drop_duplicates()
   
    
    for i in trips:
#    i=10217622870942699
   
        data_filter =  infoStopTimes[infoStopTimes['trip_id'] == i]
        nodes2 = data_filter["stop_id"]
        nodes2=nodes2.drop(nodes2.index[0])
       
        nodes1=data_filter["stop_id"]
        Count_Row=nodes1.shape[0] - 1 #gives number of row count
        nodes1=nodes1.drop(nodes1.index[Count_Row])
        
        nodes1 = pd.DataFrame(nodes1)
        dupla=nodes1.reset_index()
        dupla=dupla.drop(labels="index", axis=1)
        nodes2 = pd.DataFrame(nodes2)
        nodes2=nodes2.reset_index()
        nodes2=nodes2.drop(labels="index", axis=1)
        dupla.columns = ['node1']
        nodes2.columns = ['node2']
      
        nodes1=nodes1.reset_index()
        nodes1=nodes1.drop(labels="index", axis=1)
    
    
        for j in nodes1["stop_id"]:
            dupla['node1']=j
            dupla=dupla.join(nodes2)
            dupla = dupla[dupla.node2 != j] #Elimina la linea donde node1=node2
            dupla = dupla.drop_duplicates()
            #dupla=dupla.groupby(['node1', 'node2']).size().reset_index().rename(columns={0:'count'}) con pesos
            dupla.to_csv(pathFile, header=None, index=None, mode='a', sep=' ')
            dupla=dupla.drop(labels="node2", axis=1)
      

def ObtenerSpaceR():
      
    fileName = '..\datos\SpaceR.txt'
    
    pathFile = os.getcwd() + "\\" + fileName
                        
                       

    if path.exists(pathFile):
      os.remove(pathFile)
    
    infoStopTimes, tripsID, infoStops = LeerFicheros()
    
    #colNames = ('route_id','trip_id','stop_id')
   
    masterDF = pd.DataFrame([[0,0,0]], columns = ['trip_id','route_id', 'stop_id'], dtype=int)


     
    #print("Grouping routes")
    group_route = tripsID.groupby('route_id')
    for nombre, datos in group_route:
 
       #print(nombre)
       trips=datos
       trips=trips.drop(labels="service_id", axis=1)
       trips=trips.drop(labels="trip_headsign", axis=1)
       trips=trips.drop(labels="trip_short_name", axis=1)
       trips=trips.drop(labels="direction_id", axis=1)
       trips=trips.drop(labels="shape_id", axis=1) 
       trips = trips.drop_duplicates()
       
      # print("Grouping trips")
       group_trip_id = infoStopTimes.groupby('trip_id')
       for nombre, datos in group_trip_id: 
           datos=datos.drop(labels="arrival_time", axis=1) 
           datos=datos.drop(labels="departure_time", axis=1) 
           datos=datos.drop(labels="stop_sequence", axis=1) 
           datos=datos.drop(labels="stop_headsign", axis=1)
           datos=datos.drop(labels="shape_dist_traveled", axis=1) 
           

    
           result=pd.merge(trips, datos, on='trip_id')
           #print(result)
           Count_Row=result.shape[0]
           if Count_Row > 0:
               masterDF = masterDF.append(result)
             
      
    todosdatos = pd.DataFrame([[0,0]],columns = ['node1','node2'], dtype=int)         
    #Para componer el fichero  
    masterDF=masterDF.drop(labels="trip_id", axis=1) 
    masterDF = masterDF.drop_duplicates()
    group_stops = masterDF.groupby('stop_id')
    for nombre, datos in group_stops:
        
        nodes2=datos["route_id"]
        nodes2 = pd.DataFrame(nodes2)
        nodes2.columns = ['node2']
        nodes2=nodes2.reset_index()
        nodes2=nodes2.drop(labels="index", axis=1)
        
        nodes1=nodes2["node2"]
        nodes1 = pd.DataFrame(nodes1)
        nodes1.columns = ['node1']
        
        dupla=nodes1.reset_index()
        dupla=dupla.drop(labels="index", axis=1)
        dupla.columns = ['node1']
          
        for j in nodes1["node1"]:
            dupla['node1']=j
            dupla=dupla.join(nodes2)
            
            dupla = dupla[dupla.node2 != j] #Elimina la linea donde node1=node2
            Count_Row=dupla.shape[0]
            if Count_Row > 0:
                todosdatos = todosdatos.append(dupla)
                
            dupla=dupla.drop(labels="node2", axis=1)
          
    todosdatos = todosdatos.drop_duplicates()
    #todosdatos=todosdatos.groupby(['node1', 'node2']).size().reset_index().rename(columns={0:'count'}) pesos
   
    todosdatos = todosdatos[(todosdatos.node2 != 0)&(todosdatos.node1 != 0)]
   
    todosdatos.to_csv(pathFile, header=None, index=None, mode='w', sep=' ')
    
def ComprobarInfeccion (N, p):
    
    if (np.random.binomial(N,p) > 0):
        return INFECTADO
    else:
        return NO_INFECTADO
      
def ComprobarRecuperacion (p):
    
    if (np.random.binomial(1,p) > 0):
        return NO_INFECTADO
    else:
        return INFECTADO 

def DibujarVariasGraficas(Dir,pC, pR):
    
    
    #mi_path = os.getcwd() + "\\" + "..\\datos\\" + str(Dir)
    mi_path=Dir
    
     
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.title('Failure propagation')
    ax.set_xlabel('n')
    ax.set_ylabel('Infection rate')
    fmt = '%.5f' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(xticks)

    datosHeader = ['iter', 'rate']
    for NomFich in listdir(mi_path):
        pathFile = mi_path + "\\" + NomFich
        datos = pd.read_table(pathFile, 'engine=python', delimiter=' ', header=0, encoding = "ISO-8859-1", names=datosHeader)
        ax.plot(datos['iter'], datos['rate'], marker='o', label='pI: '+str(pC) + ' pR: '+str(pR))
        ax.legend()
    fileFig="TodosGraficos.png"
    pathFileFig = mi_path + "\\" + fileFig
    fig.savefig(pathFileFig)
    
def VerificarInfeccionVecinos(i):

    global NodeInfo
    
    Neighbours= V.neighbors(i)
    NumNodosInfect= NodeInfo.status[(NodeInfo['node'].isin (Neighbours)) & (NodeInfo['status']==INFECTADO)].sum()
    NodeInfo.status[NodeInfo.node==i] = ComprobarInfeccion (NumNodosInfect, pC)

def AplicarRecuperacion(i):

    global NodeInfo
	
    NodeInfo.status[(NodeInfo['node']==i)]=ComprobarRecuperacion(pR)
	
def EjecutarPropagacionInfeccionTime(Time):

    global Rate, TotalNodes
    
    print("Time "+ str(Time))
         
	#Propago infeccion
	#Agrupo nodos no infectados
    
    [VerificarInfeccionVecinos(i) for i in NodeInfo.node[NodeInfo.status == NO_INFECTADO]]
      
    
    TotalNodesInfect = NodeInfo.status[NodeInfo.status==INFECTADO].sum()
         
    #Aplico recuperacion
    
    [AplicarRecuperacion(i) for i in NodeInfo.node [NodeInfo.status==INFECTADO]]                
    Rate=TotalNodesInfect/TotalNodes
    
    if (Rate==0.90):
        print("Infeccion: " + str(Rate) + ", n: "+ str(Time))
         
    DatosTime = pd.DataFrame([[0,0]], columns = ['time','rate'], dtype=int)
         
    DatosTime['time']=Time
    DatosTime['rate']=Rate
             
    DatosTime.to_csv(pathFile, header=None, index=None, mode='a', sep=' ', decimal='.')
             

    DatosTime=DatosTime.drop(labels="time", axis=1)
    DatosTime=DatosTime.drop(labels="rate", axis=1)
    
def ObtenerPrimerNodo (G, Tipo, Iter):
    
    g=copy.deepcopy(G)
    
    if (Tipo == 'A'):
        nodos=g.nodes()
        FirstNode=nodos[randint(1, len(nodos))-1]
    else:
        Bet, Deg, Pag, Eig, Alea, Clustering = ObtenerPropiedadesGrafo(g)
        
        if (Tipo == 'B'):
            nodos= ObtenerNodos(Bet)
        elif (Tipo == 'D'):
            nodos= ObtenerNodos(Deg)
        elif (Tipo == 'P'):
            nodos= ObtenerNodos(Pag)
        elif (Tipo == 'C'):
            nodos= ObtenerNodos(Clustering)
        else:
            nodos= ObtenerNodos(Eig)
        #FirstNode = nodos[Iter-1] cambio para simular el mayor
        FirstNode = nodos[0]
        
        print("+++First node+++")
        print(FirstNode)
        
    return FirstNode

def EjecutarIteracionInfeccion(Iter, Tipo):

    global pathFile, Time, NodeInfo, Rate, V, TotalNodes

    print("Iteracion" + str(Iter))
    #print(Iter)
    
    print("Propagando infeccion")
     
    Nodes=V.nodes()
    TotalNodes = len(Nodes)
    print("Total nodes " + str(TotalNodes))
    FirstNode = ObtenerPrimerNodo (V, Tipo, Iter)
    #FirstNode=Nodes[randint(1, len(Nodes))-1]
    print("Primer nodo infectado")
    print(FirstNode)
    NodeInfo = pd.DataFrame(Nodes)
    NodeInfo.columns = ['node']
    NodeInfo['status'] = NO_INFECTADO
    NodeInfo.status[NodeInfo.node==FirstNode] = INFECTADO
    TotalNodesInfect = NodeInfo.status[NodeInfo.status==INFECTADO].sum()
     
    Rate = TotalNodesInfect/TotalNodes
    
    pathFile = str(NDir)+ "\\" + fileName + "_" + Tipo+ "-"+str(FirstNode)+"-"+str(Iter)+"-"+str(pC) + "_"+ str(pR)
    pathFile=pathFile.replace('.txt', '')
    pathFile= pathFile + ".txt"
    print("Directorio"+str(pathFile))
    #if (path.exists(pathFile)):
    #    os.remove(pathFile)   
		
    [EjecutarPropagacionInfeccionTime(Time) for Time in range(1, MAX_TIME+1)]
     
    print("Rate "+str("{0:.4f}".format(Rate)))
    print("End time "+str(Time))
   
       
def PropagarInfeccion2 (fileName, pR, pC, Tipo):

     global  NDir
    
     print("Creando directorio")
     NDir= os.getcwd() + "\\" + "..\\datos\\Propagacion"+str(pC)+"_"+str(pR)
     if (path.exists(NDir)):
         print("")
     else:
        #shutil.rmtree(NDir)
        os.mkdir(NDir)
     print("Ejecutando iteraciones")
     
     if (Tipo == 'A'):
         [EjecutarIteracionInfeccion(Iter, 'A') for Iter in range(1,MAX_ITER+1)]  
     elif (Tipo == 'B'):
         [EjecutarIteracionInfeccion(Iter, 'B') for Iter in range(18,MAX_ITER+1)]   
     elif (Tipo == 'D'):
         [EjecutarIteracionInfeccion(Iter, 'D') for Iter in range(1,MAX_ITER+1)] 
     elif (Tipo == 'P'):
         [EjecutarIteracionInfeccion(Iter, 'P') for Iter in range(1,MAX_ITER+1)] 

     elif (Tipo == 'C'):
         [EjecutarIteracionInfeccion(Iter, 'C') for Iter in range(17,MAX_ITER+1)]  		 
     else:
         [EjecutarIteracionInfeccion(Iter, 'E') for Iter in range(1,MAX_ITER+1)]   

  
def PropagarInfeccionTodasProbabilidades(i, Tipo):
  
    global pC, pR
    
    pC=i*0.1
    pR=0.05
    print("Probabilidad contagio " + str(pC))
 
    PropagarInfeccion2 (fileName, pR, pC, Tipo)
    
def CalcularkCores(G):
 
	# Get cores
	print("Calculating core number")
	cores = nx.core_number(G)
	# Sort cores from greater to smaller
	print("Sorting cores")
	#cores_sorted = OrderedDict(sorted(cores.items(), key=lambda t: t[1], reverse=True)) 
	print(cores)
         
 
 
def CalcularCGAleatorio(flagNodos, fileName, Iter):
#Aleatorio
   
    print("Construyendo grafo")
 
    G = ConstruirGrafo(fileName)
    Grafo = copy.deepcopy(G)
    MaxNodes = len(Grafo.nodes())
    MaxEdges = len(Grafo.edges())
	
    if flagNodos:
	
        Nodes = Grafo.nodes()
        Alea=[]
        for i in range (1, MaxNodes+1):
            Node=randint(1, MaxNodes)
            Alea.append(Nodes[Node-1])
            Alea=list(set(Alea))
	
        pref = "CGAlea"
        pathFile = os.getcwd() + "\\" + "..\\datos\\" + pref + str(Iter) + fileName
                        
        print(pathFile)
        if path.exists(pathFile):
            os.remove(pathFile)
        CalcularAislamiento (Grafo, MaxNodes, pathFile, Alea, False)  
		

    else:
            
            
        pref = "CGAleaE"
        pathFile = os.getcwd() + "\\" + "..\\datos\\" + pref + str(Iter) + fileName
                        
        print(pathFile)
        if path.exists(pathFile):
            os.remove(pathFile)
        ElemBloq=5
        results = []
        edges = random.sample(Grafo.edges(), MaxEdges)
        NumBloq= int(MaxEdges/ElemBloq)
       
        for edge in edges:
            results.append(edge)
            if (len(results) == MaxEdges):
                break
        CalcularCG (Grafo, pathFile, 0)
        for i in range(1, NumBloq+1): 
            Grafo = copy.deepcopy(G)
            E= results[0:i*ElemBloq]
            Grafo.remove_edges_from(E)
            CalcularCG (Grafo, pathFile, len(E) )	
        

def CalcularCGEDirigidos(fileName):
    
    print("Construyendo grafo")
 
    G= ConstruirGrafo(fileName)
    Grafo = copy.deepcopy(G)
   
    MaxEdges = len(Grafo.edges())
    ElemBloq=5
    NumBloq= int(MaxEdges/ElemBloq)
    
    pref = "CGTargetE"
    pathFile = os.getcwd() + "\\" + "..\\datos\\" + pref + fileName
                        
    print(pathFile)
    if path.exists(pathFile):
        os.remove(pathFile)
   
    edges = Grafo.edges()
    #print("***edges***")
    #print(edges)
   
        
    NumBloq= int(MaxEdges/ElemBloq)
    #print("*** Number of blocks ***")
    #print(NumBloq)
    
    listWeights=[]
    
    
    count = 1	
    for edge in edges: 
        try:
            print(count)
            count=count+1
            Grafo.remove_edge(edge[0], edge[1])
            hops= nx.shortest_path_length(Grafo,source=edge[0],target=edge[1])
            listWeights.append(hops)
            #Grafo = ConstruirGrafo(fileName)  
            Grafo = copy.deepcopy(G)
        except nx.NetworkXNoPath:
            print(count)
            count=count+1
            listWeights.append(99999999)
            #Grafo = ConstruirGrafo(fileName) 
            Grafo = copy.deepcopy(G)
            
    
    df1 = pd.DataFrame(edges)
    
    df2 = pd.DataFrame(listWeights)
    
    result = pd.concat([df1, df2], axis=1, ignore_index=True)
    result.columns = ['node1', 'node2', 'weight']
    
    result=result.sort_values('weight', ascending = False) 
    result.drop(['weight'], axis = 1, inplace=True)
    results=result.apply(tuple, axis=1).tolist()
   
    
            
    #Grafo = ConstruirGrafo(fileName) 
    print("...Calculando CG...")
    Grafo = copy.deepcopy(G)      
    CalcularCG (Grafo, pathFile, 0)
    for i in range(1, NumBloq+1):      
        E= results[0:i*ElemBloq]
        #Grafo = ConstruirGrafo(fileName)
        Grafo = copy.deepcopy(G)
        #print("Probando")
        #print(E)
        Grafo.remove_edges_from(E)
        CalcularCG (Grafo, pathFile, len(E) )	
        
def Menu ():
    
    while True:
        os.system('cls')
        print("Choose an option:")
        print("1. Propagation with infection starting in node with high betweeness")
        print("2. Propagation with infection starting in node with high pagerank")
        print("3. Propagation with infection starting in node with high degree")
        print("4. Propagation with infection starting in node with high eigenvector")
        print("5. Propagation with infection starting in a random node")
        print("6. Calculate robustness")
        print("7. Calculate gigant component removing nodes randomly")
        print("8. Calculate gigant component removing edges randomly")
        print("9. Calculate gigant component removing edges with high weigh")
        print("10. Calculate k-core")
        print("11. Propagation with infection starting in node with high clustering")
        print("12. Exit")
       
        option = input("")
        
        if (option =='1') or (option =='2') or (option =='3') or (option == '4') or (option == '5') or (option =='6') or (option =='7') or (option =='8') or (option =='9') or (option =='10') or (option =='11'):
            break
        
    return option
        
        
#print("Calculating robustness SpaceL")
#ObtenerSpaceL()   
#ObtenerSpaceP()      
#ObtenerSpaceR()
#g = ConstruirGrafo('..\datos\SpaceL.txt')
#ConstruirGrafo('..\datos\SpaceP.txt')
#ConstruirGrafo('..\datos\SpaceR.txt')


fileName="SpaceL.txt"
start_time = time.time() 




option = Menu()  
option = int(option)
if (option == 1):
    V = ConstruirGrafo(fileName)
    [PropagarInfeccionTodasProbabilidades(k, 'B') for k in range(1, 2)]
elif (option == 2):
    V = ConstruirGrafo(fileName)
    [PropagarInfeccionTodasProbabilidades(k, 'P') for k in range(1, 2)]
elif (option == 3):
    V = ConstruirGrafo(fileName)
    [PropagarInfeccionTodasProbabilidades(k, 'D') for k in range(1, 2)]
elif (option == 4):
    V = ConstruirGrafo(fileName)
    [PropagarInfeccionTodasProbabilidades(k, 'E') for k in range(1, 2)]
elif (option == 5):
    V = ConstruirGrafo(fileName)
    [PropagarInfeccionTodasProbabilidades(k, 'A') for k in range(1, 2)]
elif (option == 6):
    CalcularRobustez(fileName)
elif (option == 7):
    [CalcularCGAleatorio(1, "SpaceL.txt", k) for k in range(1, 101)]
elif (option == 8):
    [CalcularCGAleatorio(0, "SpaceL.txt", k) for k in range(1, 101)]
elif (option == 9):
    CalcularCGEDirigidos(fileName)
elif (option == 10):  
     V = ConstruirGrafo(fileName)
     CalcularkCores(V)
elif (option == 11):
    V = ConstruirGrafo(fileName)
    [PropagarInfeccionTodasProbabilidades(k, 'C') for k in range(1, 2)]
    
elapsed_time = time.time() - start_time
print(elapsed_time)

#DibujarVariasGraficas(os.getcwd() + "\\" + "..\\datos\\Propagacion0.1_0",0,0.1)    




#






