from Perceptron import Perceptron
from MLP import Layer
from metrics import accuracy, precision

class Adaline(Perceptron):

    def __init__(self, input_dimension):
        super().__init__(input_dimension, lambda x: x)
        self.weights_gradient = [0 for i in range(input_dimension + 1)]

class AdalineLayer(Layer):

    def __init__(self, dimension=None, input_dimension=None, activation_function=None, adaline_list=[]):
        if adaline_list == []:
            self.dimension = dimension
            self.neurons = [Adaline(input_dimension, activation_function) for i in range(dimension)]
            #self.weight_gradient = [[0 for i in range(input_dimension + 1)] for j  in range(dimension)]
        else:
            self.dimension = len(adaline_list)
            self.neurons = adaline_list
            #self.weight_gradient = [[0 for i in range(len(self.neurons[0].weights))] for j  in range(self.dimension)]

    def in_minimum(self):
        for i in range(self.neurons):
            if not all([gradient_component == 0 for gradient_component in self.neurons[i]]):
                return False

        return True

    #TO DO: CREATE AND MANTAIN DELTA WEIGHT MATRIX. THAT WILL ALLOW YOU TO HAVE A
    #A BREAK CONDITION
    def train_layer(self, dataset, epochs, learning_rate, verbose=False, save_weights=""):

        dataset.add_bias_term()
        assert(dataset.feature_vector_length() == len(self.neurons[0].weights))

        labels_header = ",".join(["prec. label " + str(key) for key in dataset.get_labels()])       
        print("Training information\n")
        print(f'epoch, accuracy, {labels_header}')

        for current_epoch in range(epochs):

            error_number = 0
            true_positives = {}
            false_positives = {}

            for key in dataset.get_labels():
                true_positives[key] = 0
                false_positives[key] = 0

            for features, expected_value in dataset:

                output_value = self.output(features)

                index = dataset.get_label_index(expected_value)

                is_incorrect = False

                for i in range(len(output_value)):

                    if i == index:
                        if output_value[index] != 1:
                            is_incorrect = True
                            self.neurons[index].adjust_weights(1, 0, learning_rate, features)
                    else:
                        if output_value[i] != 0:
                            is_incorrect = True
                            self.neurons[i].adjust_weights(0, 1, learning_rate, features)

                            if sum(output_value) == 1:
                                false_positives[str(i)] += 1

                if is_incorrect:
                    error_number += 1
                else:
                    true_positives[str(index)] += 1

            precision_list = []

            for key in dataset.get_labels():
                precision_list.append(round(precision(true_positives[key], false_positives[key]), 4))

            precision_string = ",".join([str(value) for value in precision_list])

            print(f'{current_epoch}, {accuracy(dataset.size(), error_number)}, {precision_string}')
            if not self.in_minimum():
                dataset.shuffle_all()
            else:
                break