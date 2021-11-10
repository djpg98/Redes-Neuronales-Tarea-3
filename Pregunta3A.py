import sys
from Adaline import Adaline
from Dataset import PolinomialDataset

"""El primer argumento es el grado del polinomio y el segundo el número de epochs"""
degree = int(sys.argv[1])
epochs = int(sys.argv[2])
try:
    output_file = sys.argv[3]
except:
    output_file = ''

dataset_train = PolinomialDataset('interpolacionpolinomial_train.csv', degree)
predictor = Adaline(degree)
predictor.train(dataset_train, epochs, 0.001, True, output_file)
dataset_test = PolinomialDataset('interpolacionpolinomial_test.csv', degree)
predictor.eval(dataset_test)
