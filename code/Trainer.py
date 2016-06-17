# -*- coding: utf-8 -*-


from numpy import *


class Trainer:

    def __init__(self,neuralNetwork,backPropagation):
        self.neuralNetwork = neuralNetwork
        self.backPropagation = backPropagation

    def train(self,input,target, iterations=1000,N=0.3):

        print ("---------- Errors ----------\n")
        # Número de iteracciones
        for i in range(iterations):
            error = 0.0

            for p in range(size(input,axis = 0)): # Axis= 0 --> Cuenta las filas
                inputs = input[p,:]
                targets = target[p,:]
                self.neuralNetwork.evaluar(inputs)
                error = error + self.backPropagation.backpropagation(targets,N)

            if i%100 == 0:
                print("Iteration: {0}  Error:{1}".format(i,error))

        #El error obtenido en la ultima iteracción
        print ("Final error:{0} \n---------------------------  \n".format(error))
        #La matriz de pesos
        print ("Weight matrix:{0} \n---------------------------  \n".format(self.neuralNetwork.weights))