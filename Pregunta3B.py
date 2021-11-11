import sys
from Adaline import Adaline
from Dataset import PolinomialDataset

"""El primer argumento es el grado del polinomio y el segundo la tasa de aprendizaje"""
degree = int(sys.argv[1])
learning_rate = float(sys.argv[2])
print(learning_rate)
try:
    output_file = sys.argv[3]
except:
    output_file = ''

dataset_train = PolinomialDataset('interpolacionpolinomial_train.csv', degree)
predictor = Adaline(degree)
predictor.train(dataset_train, 100, learning_rate, True, output_file)
dataset_test = PolinomialDataset('interpolacionpolinomial_test.csv', degree)
predictor.eval(dataset_test)
dataset_real = PolinomialDataset('interpolacionpolinomial_real.csv', degree)
predictor.eval(dataset_real)
