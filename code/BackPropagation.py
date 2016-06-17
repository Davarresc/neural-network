# -*- coding: utf-8 -*-

from numpy import *


class BackPropagation:

    def __init__(self,neuralNetwork):
        self.neuralNetwork = neuralNetwork


    #Función derivada de sigmoide
    def dsigmoid(self,val):
        return val - val*val


    ''''''''
    '''
    '''''''''

    def backpropagation(self,targets,N):

        error = sum(0.5*(targets-self.neuralNetwork.ao)**2)

        #Calculamos el error cometido en la salida
        difp = self.dsigmoid(self.neuralNetwork.ao)*(targets - self.neuralNetwork.ao) # yi - ai
        difp_aux = difp # Se usa luego para calcular la matriz de pesos, ya que esta no depende del nuevo delta calculado, si no de este

        #Calculamos el delta de la ultima capa oculta
        difp = self.dsigmoid(self.neuralNetwork.ah[len(self.neuralNetwork.ah)-1])*dot(self.neuralNetwork.weights[len(self.neuralNetwork.weights)-1],difp_aux)

         # Calculamos la matriz de peso de la ultima capa oculta a la capa de salida
        self.neuralNetwork.weights[len(self.neuralNetwork.weights)-1] = self.neuralNetwork.weights[len(self.neuralNetwork.weights)-1]\
                + N * difp_aux * reshape(self.neuralNetwork.ah[len(self.neuralNetwork.ah)-1],(self.neuralNetwork.ah[len(self.neuralNetwork.ah)-1].shape[0],1))

        for i in range(len(self.neuralNetwork.ah)-1,0,-1):
            difp_aux = difp
            difp = self.dsigmoid(self.neuralNetwork.ah[i-1])*dot(self.neuralNetwork.weights[i],difp_aux)

            self.neuralNetwork.weights[i] = self.neuralNetwork.weights[i] + N * difp_aux * reshape(self.neuralNetwork.ah[i-1],(self.neuralNetwork.ah[i-1].shape[0],1))


        self.neuralNetwork.weights[0] = self.neuralNetwork.weights[0] + N * difp * reshape(self.neuralNetwork.ai,(self.neuralNetwork.ai.shape[0],1))

        #Devolver error
        return error