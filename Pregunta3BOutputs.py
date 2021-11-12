import sys
from Adaline import Adaline
from Dataset import PolinomialDataset

"""El primer argumento es el grado del polinomio y el segundo el archivo donde se guardaran los outputs"""
degree = int(sys.argv[1])
try:
    output_file = sys.argv[2]
except:
    output_file = ''

predictor = Adaline(degree)
predictor.weights = [2.534265838416539, -0.28065266148619816, 0.5132917165492941, -0.21148255570672153, 0.18700927751083102, 1.7904283297718628e-05, -0.21230726664727806, 0.34006087186208334]
dataset_real = PolinomialDataset('interpolacionpolinomial_real.csv', degree)
predictor.eval(dataset_real, output_file)