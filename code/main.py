# -*- coding: utf-8 -*-

from Test import *

def get_letter(path, array):
    path = path.format(i)
    pgm = Pgm(path,20)
    pgm.read_pgm_file();
    #Datos entrada compuerta
    array.append(pgm.result)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''  Parameters to configure
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
inputT = []
targetT = []
valuate = []
targetE = []

for i in range(29):
    get_letter("letra a/{0}.pgm",inputT)
    targetT.append([0,0,0,0,1])
    get_letter("letra e/{0}.pgm",inputT)
    targetT.append([0,0,0,1,0])
    get_letter("letra j/{0}.pgm",inputT)
    targetT.append([0,0,1,0,0])
    get_letter("letra g/{0}.pgm",inputT)
    targetT.append([0,1,0,0,0])
    get_letter("letra m/{0}.pgm",inputT)
    targetT.append([1,0,0,0,0])

for i in range(i,30):
    get_letter("letra a/{0}.pgm",valuate)
    targetE.append([0,0,0,0,1])
    get_letter("letra e/{0}.pgm",valuate)
    targetE.append([0,0,0,1,0])
    get_letter("letra j/{0}.pgm",valuate)
    targetE.append([0,0,1,0,0])
    get_letter("letra g/{0}.pgm",valuate)
    targetE.append([0,1,0,0,0])
    get_letter("letra m/{0}.pgm",valuate)
    targetE.append([1,0,0,0,0])


#Values
inputsTraining = array(inputT)
targetTraining = array(targetT)
inputsEvaluting = array(valuate)
targetEvaluting = array(targetE)

#Neural network
inputNodes = 900
hiddenLayers = (60,)

outputNodes = 5

# Maximun and minimun value for wieght matrix
maxVal = 0.5
minVal = -0.5

doTraining = True # If it is true it will do the training, in other case it loads the old matrix.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''  End parameters to configure
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

weightMatrices = Test.get_weight_matrices(inputNodes,hiddenLayers,outputNodes,maxVal,minVal)

test = Test(inputsTraining,targetTraining,inputsEvaluting,targetEvaluting,20,0.1)

test.test(inputNodes,hiddenLayers,outputNodes,weightMatrices,doTraining)
