# -*- coding: utf-8 -*-

from numpy import *


class NeuralNetwork:

    def __init__(self,inputNodes,hiddenLayers,outputNodes,weigthMatrices):
        self.inputNodes = inputNodes + 1 #Entradas + bía para sesgo estadístico
        self.hiddenLayers = hiddenLayers
        self.outputNodes = outputNodes
        self.weights = weigthMatrices


        #Salida de las neuronas

        self.ai =  ones((self.inputNodes)) #Salida capa entrada
        self.ai[len(self.ai)-1]=-1 #Valor de la bia

        self.ah = list(ones((i+1)) for i in hiddenLayers) #Salida por cada capa oculta

        for i in range(len(self.ah)-1):
            self.ah[i][self.hiddenLayers[i]] = -1

        self.ah[len(self.hiddenLayers)-1] = ones((self.hiddenLayers[len(self.hiddenLayers)-1]))
        #Salida de la ultima capa oculta sin bias


        self.ao = ones((self.outputNodes))  #Salida capa salida


    # Función sigmoide
    def sigmoid(self,x):
        x = 1/(1+e**(-x))
        return x

    ''''''''
    ''' Dada una entrada devuelve una salida con los datos calculados según la propagación hacia adelante
    '''''''''
    def evaluar(self,inputs):

       #Valores de entrada
        self.ai[0:self.inputNodes-1] = inputs

       #Valores salida de las capas ocultas
        in_i = dot(transpose(self.weights[0]),self.ai)
        self.ah[0] = self.sigmoid(in_i)
        self.ah[0][-1]=-1

        for i in range(len(self.hiddenLayers)-1):
            in_i = dot(transpose(self.weights[i+1]),self.ah[i])
            self.ah[i+1] = self.sigmoid(in_i)
            self.ah[i+1][-1] = -1

       #Valores de salida
        Wn_an = dot(transpose(self.weights[len(self.weights)-1]),self.ah[len(self.ah)-1])
        self.ao = self.sigmoid(Wn_an)

        return self.ao