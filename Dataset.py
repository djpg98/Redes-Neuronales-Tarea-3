import csv
import random
import math

"""" Dado un valor x y un entero degree, devuelve un vector de la forma 
     [1, x^1, x^2,...,x^degree]
     Parámetros:
        - x: Valor que servirá como base para las potencias en este vector
        - degree: Máxima potencia a inlcuir en el vector. Se incluyen todas
        las potencias que van de 0 a degree en ese orden.
"""
def powers_of_x(x, degree):
    return [math.pow(x, i) for i in range(degree + 1)]

""" Conjunto de funciones básicas que debe tener un dataset """
class DatasetMixin:

    """ Añade una componente de sesgo a cada dato en el dataset. Esto se logra
        agregando una componente adicional que siempre vale 1 a todos los datos 
        en el dataset.
    """
    def add_bias_term(self):

        for i in range(len(self.features)):

            self.features[i].insert(0, 1)

    """ Devuelve la cantidad de elementos en el dataset """
    def size(self):

        return len(self.features)

    """ Devuelve la cantidad de elementos pertenecientes
        al conjunto de entrenamiento del dataset """
    def training_data_size(self):

        return len(self.training_data)

    """ Devuelve la cantidad de elementos pertenecientes
        al conjunto de prueba del dataset """
    def test_data_size(self):

        return len(self.test_data)

    """ Devuelve el tamaño del input vector (Incluendo el término de bias si
        este ha sido agregado
    """
    def feature_vector_length(self):

        return len(self.features[0])

    """ Iterador para todos los elementos del dataset """
    def __iter__(self):

        for pair in zip(self.features, self.values):

            yield pair

    """ Iterador para el conjunto de datos de entrenamiento
        del dataset
    """
    def training_data_iter(self):

        for index in self.training_data:

            yield (self.features[index], self.values[index])

    """ Iterador para el conjunto de datos de prueba
        del dataset
    """
    def test_data_iter(self):

        for index in self.test_data:

            yield (self.features[index], self.values[index])

    """ Altera aleatoriamente el orden en que se iteran los elementos del
        conjunto de datos de entrenamiento
    """
    def shuffle_training_data(self):
        random.shuffle(self.training_data)

""" Esta clase representa a los datasets donde solo existen dos categorías
    Una será representada con la etiqueta 1 (La llamada categoría positiva)
    y las demás con 0. La clase utiliza los atributos training_data y 
    test_data para determinar los índices de los elementos de la lista que
    contiene el dataset que serán utilizados en los conjuntos de entrenamiento
    y de prueba. Nótese que shuffle_training_data no altera el orden de los
    elementos en el dataset, solo el de los índices en training_data
"""
class BinaryDataset(DatasetMixin):

    """ Constructor de la clase
        Parámetros:
            - Datafile: El archivo con los datos en formato csv.
            - positive_category: La categoría que utilizará la etiqueta 1
    """
    def __init__(self, datafile, positive_category):

        self.features = []
        self.values = []

        with open(datafile, 'r') as csv_file:

            data_reader = csv.reader(csv_file, delimiter=",")

            #SKip header
            next(data_reader)

            for row in data_reader:

                features, value = row[:-1], row[-1:][0]

                if positive_category == value:
                    numeric_value = 1
                else:
                    numeric_value = 0

                self.features.append(list(map(float, features)))
                self.values.append(numeric_value)

            csv_file.close()

        index_list = [i for i in range(len(self.features))]
        self.training_data = random.sample(index_list, int(0.80 * len(self.features)))
        self.test_data = [index for index in index_list if index not in self.training_data]

""" Esta clase se utiliza para datasets de múltiples categorías.
    En la etiqueta se guarda el ínidce del perceptron en la lista de
    perceptrones de la clase Layer que es el encargado de reconocer
    dicha categoría. Utiliza el atributo index_list para determinar
    el orden en que se iterará el dataset
"""
class MultiClassDataset(DatasetMixin):

    """ Constructor de la clase.
        Parámetros:
            - datafile: El archivo con los datos en formato csv
            - label_dictionary: Un diccionario que contiene el
            índice del perceptrón encargado de reconocer la categoría
            con el label indicado
    """
    def __init__(self, datafile, label_dictionary):

        self.features = []
        self.values = []
        self.label_dictionary = label_dictionary

        with open(datafile, 'r') as csv_file:

            data_reader = csv.reader(csv_file, delimiter=",")
            
            #SKip header
            next(data_reader)

            for row in data_reader:

                features, value = row[1:], row[0]

                self.features.append(list(map(float, features)))
                self.values.append(value)

            csv_file.close()

        self.index_list = [i for i in range(len(self.features))]
        self.training_data = random.sample(self.index_list, int(0.80 * len(self.features)))
        self.test_data = [index for index in self.index_list if index not in self.training_data]
        self.shuffle_all()

    """ Utiliza una función para normalizar los datos
        - normalizer_function: Función utilizada para normalizar 
    """
    def normalize_data(self, normalizer_function):

        for i in range(len(self.features)):
            self.features[i] = list(map(normalizer_function, self.features[i]))

    """ Dada la etiqueta de la categoría, devuelve el índice del perceptrón
        encargado de reconocerla
        - label: Etiqueta de la categoría
    """
    def get_label_index(self, label):
        return self.label_dictionary[label]

    def get_label_vector(self,label):
        vector = [-1 for i in range(len(self.label_dictionary))]
        vector[self.get_label_index(label)] = 1
        return vector

    """ Iterar por todas las etiquetas (Categorías) posibles"""
    def get_labels(self):

        for key in self.label_dictionary.keys():
            yield key

    """ Altera aleatoriamente el orden de los índices en index_list,
        lo que permite que se pueda iterar en orden distinto en
        distintas epochs"""
    def shuffle_all(self):
        random.shuffle(self.index_list)

    """" Iterador por todos los elementos del dataset"""
    def __iter__(self):

        for index in self.index_list:
            yield (self.features[index], self.values[index])
            
""" Esta Dataset se utiliza cuando se tiene un punto x y otro y se desea 
    conseguir un interpolador polinomial que permita obtener y a partir de x"""
class PolinomialDataset(DatasetMixin):

    """ Constructor de la clase
        Parámetros:
            - datafile: El archivo con los datos en formato csv
            - degree: Grado del interpolador polinomial
    """
    def __init__(self, datafile, degree):

        self.features = []
        self.values = []

        with open(datafile, 'r') as csv_file:

            data_reader = csv.reader(csv_file, delimiter=",")
            
            #SKip header
            next(data_reader)

            for row in data_reader:

                features, value = powers_of_x(float(row[0]), degree), float(row[1])

                self.features.append(features)
                self.values.append(value)

            csv_file.close()

        self.index_list = [i for i in range(len(self.features))]
        self.shuffle_all()

    """ Altera aleatoriamente el orden de los índices en index_list,
        lo que permite que se pueda iterar en orden distinto en
        distintas epochs"""
    def shuffle_all(self):
        random.shuffle(self.index_list)

    """" En esta clase, lo que sería el componente correspondiente al sesgo se crea al momento
         de asignar crear el vector con todas las potencias del valor de entrada en el dataset,
         Para evitar errores, puse este método que te devuelve un error si intentas agregar el 
         sesgo otra vez después de eso
    """
    def add_bias_term(self):

        raise Exception('clase PolinomialDataset agregra el componente correspondiente al sesgo al momento de crear el dataset. No invocar este método')

    """" Iterador por todos los elementos del dataset"""
    def __iter__(self):

        for index in self.index_list:
            yield (self.features[index], self.values[index])
