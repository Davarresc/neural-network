# -*- coding: utf-8 -*-



import re
import numpy
from matplotlib import pyplot


'''''''''''''''''''''''''''''''''''''''''
''  Método para leer el fichero PGM
''  Formato permitido:
''          #1   Cabecera
''          #2   Fila Columna
''          #3   Dígito de control
''          #4   Datos
'''''''''''''''''''''''''''''''''''''''''


class Pgm:

    path = ""               # Fichero de la imagen a leer
    resultMatrix = []       # Matríz de los datos leídos
    result = []             # Array con los datos leídos
    threshold = 0           # Umbral a partir de qué color se considera negro (Binarización)
    maxGrayValue = 0        # Nivel máximo de gris
    rows = 0                # Número de filas
    cols = 0                # Número de columnas

    def __init__(self,path,threshold):
        self.path = path
        self.threshold = threshold


    ###
    #   Leer cabecera, nivel de gris máximo, número de filas y columnas y datos del fichero.
    ###
    def read_pgm_file(self):
        fin = None

        try:
            fin = open(self.path, 'r')

            # Obtenemos la cabecera, en nuestro caso debe ser P2
            fin = self.getHeader(fin)


            #Obtenemos las filas y las columnas de la matriz
            fin = self.getDim(fin)


            # Leemos los datos de la imagen
            result = self.getData(fin)

        finally:
            if fin != None:
                fin.close() # Cerramos el fichero
            fin = None

        return result

    def getHeader(self,fin):
        found = False
        while not found:
            header = fin.readline().strip()
            if header.startswith('#'): #Si es un comentario no se hace nada
                continue
            elif header == 'P2': #En otro caso debe ser la cabecera P2,
                found = True
            else:
                raise ValueError("Invalid header") # Si no es un comentario ni la cabecera que esperamos se finaliza

        return fin

    def getDim(self,fin):
        found = False
        while not found:
            header = fin.readline().strip()
            if not header.startswith('#'): #Si no es un comentario se analiza
                match = re.match('^(\d+) (\d+)$', header)
                if match == None:
                    raise ValueError("Invalid argument")

                cols, rows = match.groups() # Devuelve [columnas, filas]

                self.rows = int(rows)
                self.cols = int(cols)

                if (rows, cols) == (0, 0): # Si ambos son 0 no es válido
                    raise ValueError("Invalid argument")

                found = True
        return fin

    def getData(self,fin):
        self.resultMatrix = numpy.zeros((self.rows, self.cols), numpy.int8)
        self.result = numpy.zeros((self.rows*self.cols),numpy.int8)
        row = 0
        col = 0
        stop = False
        while not stop:
            line = fin.readline().strip()
            if line == '': # EOF
                stop = True
            for c in line.split(): # Separamos la línea por los espacios
                if c == ' ':
                    continue
                val = int (c)
                val = 1 if val > self.threshold else 0 # Normalizamos a 0 o 1 (Binarización)
                if self.maxGrayValue <= 0:  # Si es 0, es el dígito de control
                        self.maxGrayValue = val
                else:
                    self.resultMatrix[row,col] = val
                    self.result[self.cols*row+col] = val
                    col += 1
                if col == self.cols:
                    row += 1
                    col = 0
                if row == self.rows:
                    stop = True
        return self.result

    def show_image(self):
        image = self.resultMatrix
        pyplot.imshow(image, pyplot.cm.gray)
        pyplot.show()

    def get_matches_between_images(self,image):

        var = numpy.zeros((self.resultMatrix.shape[0],self.resultMatrix.shape[1]))
        total = 0
        matches = 0

        for i in range(self.resultMatrix.shape[0]):
            for j in range(self.resultMatrix.shape[1]):
                if self.resultMatrix[i][j]==image[i][j]:
                    var[i][j] = 1
                    matches += 1
                total +=1

        print(matches)