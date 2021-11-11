from Perceptron import Perceptron
from MLP import Layer
from metrics import accuracy, precision, sample_error
import csv

""" Subclase de la clase Perceptron con las adaptaciones necesarias para que un
    Perceptrón genérico pase a ser un Adaline. Nótese que la diferencia fundamental
    es que el Adaline utiliza la función lineal para entrenar, y si es necesario
    hacer una clasificación binaria, utiliza una función (Normalmente la umbral)
    a la hora de evaluar
"""
class Adaline(Perceptron):

    """ Método constructor de la clase. Básicamente llama al constructor de Perceptron
        y agrega dos campos: weights_gradient, que almacena la dirección contraria al
        gradiente (Y permitiría verificar si se alcanzó un mínimo) y threshold_function,
        que permite agregar una función threshold_adicional a la de activación a la 
        hora de hacer la evaluación (No el entrenamiento), si se requiriera hacer una
        clasificación binaria
        Parámetros:
            - input_dimension: Cantidad de inputs que recibe el perceptron (Sin contar el sesgo)
            - threshold_function: Función que permite hacer una clasificación binaria para un
                input durante la etapa de prueba o evaluación
    """
    def __init__(self, input_dimension, threshold_function=None):
        super().__init__(input_dimension, lambda x: x)
        self.weights_gradient = [0 for i in range(input_dimension + 1)]
        self.threshold_function=threshold_function

    """ Permite ajustar los pesos del perceptron cuando hay un dato mal clasificado 
        Parámetros:
            - expected_value: Valor que se esperaba devolviera el perceptron para el dato dado
            - output_value: Valor devuelto por el perceptron para el dato dado
            - learning_rate: Tasa de aprendizaje a aplicar
            - features: El dato a partir del cual se obtuvo el resultado de output_value
    """
    def adjust_weights(self, expected_value, output_value, learning_rate, features):

        factor = learning_rate * (expected_value - output_value)

        self.weights_gradient = list(map(lambda x: factor * x, features))

        self.weights = list(map(lambda pair: pair[0] + pair[1], zip(self.weights, self.weights_gradient)))

    """ Utiliza el output de la función de activación (La función lineal) como input de la función
        self.threshold, para poder realizar una clasificación de los datos. Esto se utiliza, por ejemplo
        al momento de hacer la evaluación del conjunto test en la pregunta 2, ya que la idea ahí era tener
        un vector de -1 y 1, por lo que hacía falta aplicar una función umbral.
        Parámtros:
            - inputs: Vector que actúa como input del Adaline 
    """
    def output_with_threshold(self, inputs):
        return self.threshold_function(self.activation_function(self.sum_inputs(inputs)))

    """ Implementación del algoritmo LMS para un único Adaline:
        Parámetros:
            - dataset: Instancia de una clase que hereda el mixin DatasetMixin (En esta tarea
              existen tres: BinaryDataset, MultiClassDataset y PolinomicalDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
            - epochs: Número máximo de epochs durante el entrenamiento
            - learning_rate: Tasa de aprendizaje
            - verbose: Si se desea imprimir información de los errores en cada epoch/pesos finales
            - save_training_error: Nombre de un archivo csv en el que se guardará el error promedio
              para cada epoch
    """
    def train(self, dataset, epochs, learning_rate, verbose=False, save_training_error=''):
        #No hace falta hacer dataset.add_bias() porque en en la pregunta 3
        #el bias se añade al momento de crear el vector de características 
        #Apartir del dato, pero si se fuera a usar otro Dataset que no
        #sea de clase PolinomicalDataset, debe agregarse aquí
        assert(dataset.feature_vector_length() == len(self.weights))

        if save_training_error != '':
            training_data = [["epoch", "error"]]

        print("Training information\n")
        print("Epoch, error")

        for current_epoch in range(epochs): #Para cada epoch

            sum_mse = 0 #Aquí se va acumulando el error para cada muestra

            for features, expected_value in dataset: #Se itera sobre las muestras en el dataset

                output_value = self.output(features) #Se produce el output dados los features (Utilizando la función lineal)
                
                error = sample_error([expected_value], [output_value]) #Se calcula el error para este sample

                self.adjust_weights(expected_value, output_value, learning_rate, features) #Se asjustan los pesos de acuerdo al gradiente

                sum_mse += error #Actualizar error total

            mse = sum_mse / dataset.size() #Dividie el error entre el número de muestras para tener el error promedio

            dataset.shuffle_all() #Cambiar el orden en que se muestran los datos
            if verbose:
                print(f'{current_epoch + 1}, {mse}')

            if save_training_error != '':
                training_data.append([current_epoch + 1, mse])


        if verbose:
            print("Pesos finales: ")
            print(self.weights)
            print("")

        if save_training_error != '': #Escribir en un archivo el error cometido en cada epoch

            with open(save_training_error, 'w') as training_results:
                writer = csv.writer(training_results)

                for row in training_data:
                    writer.writerow(row)

                training_results.close()


    """ Devuelve la precision y la accuracy para un dataset test
        Parámetros:
            - Dataset: Instancia de una clase que hereda el mixin DatasetMixin (En esta tarea
              existen tres: BinaryDataset, MultiClassDataset y PolinomicalDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
    """
    def eval(self, dataset):

        print("Test information\n")
        print("error")

        sum_mse = 0

        for features, expected_value in dataset:

            output_value = self.output(features)
            error = sample_error([expected_value], [output_value])
            sum_mse += error

        mse = sum_mse / dataset.size()
            
        print(f'{mse}')

""" Subclase de la clase Layer, pero que utiliza Adalines en vez de Perceptrons"""
class AdalineLayer(Layer):

    """ Constructor de la clase Layer. 
        Parámetros:
            - dimension: Cantidad de Adalines en la capa
            - input_dimension: Dimensiones del vector de input que recibirán los perceptrones de la capa
            - threshold_function: De ser necesaria una clasificación binaria de los datos durante la evaluación
              se debe suministrar esta función, que se encargará de eso. Normalmente es la función umbral
            - perceptron_list: En caso de no pasar ninguno de los parámetros anteriores, se presenta la
              opción de pasar directamente una lista de perceptrones. Nótese que si este parámetro se
              pasa junto con los demás, los demás serán ignorados
    
    """
    def __init__(self, dimension=None, input_dimension=None, threshold_function=None, adaline_list=[]):
        if adaline_list == []:
            self.dimension = dimension
            self.neurons = [Adaline(input_dimension, threshold_function) for i in range(dimension)]
        else:
            self.dimension = len(adaline_list)
            self.neurons = adaline_list

    """ Permite verficar si se alcanzó un mínimo verificando si las componentes del 
        gradiente se hicieron 0"""
    def in_minimum(self):
        for i in range(len(self.neurons)):
            if not all([gradient_component == 0 for gradient_component in self.neurons[i].weights_gradient]):
                return False

        return True

    """ Aplica la función threshold a todos los Adalines de la capa dado un dato y devuelve
        un vector (Representado con una lista) que contiene los resultados de cada Adaline
        Parámetros:
            - input_vector: Dato de entrada suministrado a la capa
    
    """
    def output_with_threshold(self, input_vector):

        return [adaline.output_with_threshold(input_vector) for adaline in self.neurons]

    """ Implementación del algoritmo LMS para una capa de Adalines
        Parámetros
            - dataset: Una clase que hereda el mixin DatasetMixin (En esta tarea
              existen dos: BinaryDataset y MultiClassDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
            - epochs: Número máximo de epochs durante el entrenamiento
            - learning_rate: Tasa de aprendizaje
            - verbose: Si se desea imprimir información de los errores en cada epoch/pesos finales
            - save_weights: Nombre del archivo donde se guardaran los pesos. Si el nombre es el 
              string vacío no se salvan los pesos (Esta parte no está funcionando porque no he
              implementado save_weights aquí)
    """
    def train_layer(self, dataset, epochs, learning_rate, verbose=False, save_weights=""):

        dataset.add_bias_term()
        assert(dataset.feature_vector_length() == len(self.neurons[0].weights))
       
        print("Training information\n")
        print(f'epoch, MSE')

        prev_mse = 0 #Aquí se guarda el mse de la epoch anterior

        for current_epoch in range(epochs):

            sum_mse = 0 #Aquí se va acumulando el error para cada muestra

            for features, expected_value in dataset: #Se itera sobre las muestras en el dataset

                output_value = self.output(features) #Se produce el output dados los features (Utilizando la función lineal)

                index = dataset.get_label_index(expected_value) #Se obtiene el índice que representa al Adaline encargado de reconocer esta categoría

                error = sample_error(dataset.get_label_vector(expected_value), output_value) #Se calcula el error para la muestra

                #Esta es la parte que está muy específica para la pregunta 2, ya que asume que el
                #la salida esperada en el Adaline encargado de reconocer la categoría indicada en
                #expected_value va a ser 1 y para los demás -1. Si se utiliza otra función umbral
                #o simplemente se espera un vector de enteros o reales como salida de la capa,
                #habría que modificar esta parte,aunque no sería nada muy complicado
                for i in range(len(output_value)):

                    if i == index:
                        self.neurons[index].adjust_weights(1, output_value[index], learning_rate, features)
                    else:
                        self.neurons[i].adjust_weights(-1, output_value[i], learning_rate, features)



                sum_mse += error #Actualizar error total

            mse = sum_mse / dataset.size() #Calcular error promedio

            print(f'{current_epoch}, {mse}')
            if abs(prev_mse - mse) >= 0.000001: #Criterio de parada
                prev_mse = mse 
                dataset.shuffle_all() #Cambiar el orden en que se muestran los datos
            else:
                break

    """ Devuelve la precision, la accuracy y el error cuadrático para un dataset test
        Parámetros:
            - Dataset: Instancia de una clase que hereda el mixin DatasetMixin (En esta tarea
              existen tres: BinaryDataset, MultiClassDataset y PolinomicalDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
    """
    def eval(self, dataset):

        dataset.add_bias_term()
        assert(dataset.feature_vector_length() == len(self.neurons[0].weights))

        labels_header = ",".join(["prec. label " + str(key) for key in dataset.get_labels()])
        print('Test information\n')
        print(f'accuracy, mse, {labels_header}')

        error_number = 0
        true_positives = {}
        false_positives = {}

        for key in dataset.get_labels():
            true_positives[key] = 0
            false_positives[key] = 0

        sum_mse = 0
        for features, expected_value in dataset:

            output_value = self.output_with_threshold(features)

            index = dataset.get_label_index(expected_value)

            is_incorrect = False

            error = sample_error(dataset.get_label_vector(expected_value), output_value)

            for i in range(len(output_value)):

                if i == index:
                    if output_value[index] != 1:
                        is_incorrect = True
                else:
                    if output_value[i] != -1:
                        is_incorrect = True

                        if sum(output_value) == -9:
                            false_positives[str(i)] += 1

            if is_incorrect:
                error_number += 1
            else:
                true_positives[str(index)] += 1

            sum_mse += error

        mse = sum_mse / dataset.size()

        precision_list = []

        for key in dataset.get_labels():
            precision_list.append(round(precision(true_positives[key], false_positives[key]), 2))

        print("ERROR NUMBER")
        print(error_number)
        print("SIZE")
        print(dataset.size())
        precision_string = ",".join([str(value) for value in precision_list])
        print(f'{accuracy(dataset.size(), error_number)}, {mse}, {precision_string}')
