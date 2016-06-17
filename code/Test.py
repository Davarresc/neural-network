# -*- coding: utf-8 -*-

from Pgm import *
from NeuralNetwork import *
from BackPropagation import *
from Trainer import *
from numpy import *


class Test:

    def __init__(self, inputsTraining,targetTraining,inputsEvaluting,targetEvaluting,iterations,N):
        self.inputsTraining = inputsTraining
        self.targetTraining = targetTraining
        self.inputsEvaluting = inputsEvaluting
        self.targetEvaluting = targetEvaluting
        self.iterations = iterations
        self.N = N

    def test(self,inputNodes,hiddenLayers,outputNodes,weightMatrices,doTraining=True):
        nw=NeuralNetwork(inputNodes,hiddenLayers,outputNodes,weightMatrices)

        bp = BackPropagation(nw)

        trainer = Trainer(nw,bp)

        if (doTraining):
            trainer.train(self.inputsTraining,self.targetTraining,self.iterations,self.N)
            numpy.save("weights",trainer.backPropagation.neuralNetwork.weights)
        else:
            trainer.backPropagation.neuralNetwork.weights = numpy.load("weights.npy")

        self.evaluate(trainer,self.inputsEvaluting,self.targetEvaluting)

    def evaluate(self,trainer,input,target):

        num_correct = 0;
        total = 0;
        for p in range(size(input,axis = 0)): # Axis= 0 --> Cuenta las filas
            inputs = input[p,:]
            targets = target[p,:]
            original = trainer.neuralNetwork.evaluar(inputs)
            output = zeros(len(original))
            output[original.argmax(axis=0)] = 1

            if (targets == output).all():
                num_correct += 1

            total += 1

            print ("Target: {0}  Output: {1} Original output: {2} \n".format(targets,output,original))

            if not (targets == output).all():
                print("Fallo en la prueba número {0}".format(p))


        print ("{0}/{1}={2}%".format(num_correct,total,(num_correct*1.0/total)*100))


    @staticmethod
    def get_weight_matrices(inputNodes,hiddenLayers,outputNodes,maxVal,minVal):
        #Creación matrices de peso
        weights = list()

        #random.uniform --> Filas x columnas
        weights.append(
            random.uniform(minVal,maxVal,
                           (inputNodes+1,hiddenLayers[0]+1)
            )
        )

        for i in range(len(hiddenLayers)-1):
            weights.append(random.uniform(minVal,maxVal,(hiddenLayers[i]+1,hiddenLayers[i+1]+1)))

        weights.append(
            random.uniform(minVal,maxVal,(
                hiddenLayers[len(hiddenLayers)-1]+1,outputNodes)
            )
        )

        return weights
