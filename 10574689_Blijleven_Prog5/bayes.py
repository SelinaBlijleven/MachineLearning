"""
	Selina Blijleven 	10574689
	
    Programming Assignment 5
    December 2015
"""

import csv
import numpy as np

class GaussianNaiveBayes(object):
    
    """
        Predict functions
    """
    def test_model(self, filename, misclassified=False):
        """
            Tests the model with the given file and returns the accuracy.
        """
        self.test = self.read(filename)
        
        predictions = []
        for row in self.test:
            best_pred, class_probabilities = self.predict(row[:-1])
            predictions.append(best_pred)
            self.prediction_probabilities.append(class_probabilities)
        
        acc, incorrect = self.calc_accuracy(predictions)
        
        if misclassified == True:
            self.eval_misclassified(predictions, incorrect)
        return acc
            
    def predict(self, row):
        """
            Returns the classification with the highest probability
            (given by Gaussian Naive Bayes). The probability for each
            class is stored in prediction_probabilities.
            argmax p(class) * Pi P(x = value|class)
        """
        max = 0
        max_key = None
        
        probability_class = {}
        
        for key in self.class_probs:
            values_prob = self.class_probability(key, row)
            class_prob = self.class_probs[key]
            prob = values_prob * class_prob
            
            probability_class[key] = prob
            
            if prob > max:
                max = prob
                max_key = key
            
        return max_key, probability_class
        
    def class_probability(self, class_key, row):
        """
            Uses the mean value and variance of a variable
            in combination with the probability of the classification
            to find the probability of the classification.
            Pi P(x = value|class)
        """
        product = 1
        
        for i in range(len(row)):
            var_value = row[i]
            var_mean = self.variable_means[i][class_key]
            var_variance = self.variable_variances[i][class_key]
            value_in_class = self.probability_distribution(var_value, var_mean, var_variance)
            product = product * value_in_class
            
        return product
        
    def probability_distribution(self, value, mean, variance):
        """
            Calculates the probability of the value associated with
            the class with the Gaussian distribution.
            P(x = value|class)
        """

        if variance > 0:
            a = 1 / (np.sqrt(2 * np.pi * variance))
            b = np.e**-((value - mean)**2 / (2 * variance))
            return a * b
        else:
            return 1
        
    def calc_accuracy(self, predictions):
        """
            Calculates the accuracy as a percentage of
            correct classifications. Also returns an array
            of indices of wrong classifications.
        """
        correct = 0
        incorrect = []
        set_size = len(predictions)
        
        for i in range(set_size):
            classification = self.test[i][-1]
            prediction = predictions[i]
            if classification == prediction:
                correct += 1
            else:
                incorrect.append(i)
        return (correct / float(set_size) * 100), incorrect
        
    def eval_misclassified(self, predictions, misclassified):
        """
            Assigns the relative difference between probabilities
            of the actual classification and prediction to find most
            likely misclassified examples.
        """
        misclassified_prob = []
        
        for i in misclassified:
            classification = self.test[i][-1]
            classification_prob = self.prediction_probabilities[i][classification]
            prediction = predictions[i]
            prediction_prob = self.prediction_probabilities[i][prediction]
            misclassified_prob.append([(prediction_prob / classification_prob), prediction, classification, i])
        
        misclassified_prob = sorted(misclassified_prob)
        print misclassified_prob[-10:]
                
    """
        Training functions
    """
    
    def train(self):
        """
            Makes a dictionary of every parameter and all the classes
            and assigns a probability to each value for each parameter
            or each class.
        """
        param_size = len(self.training[0]) - 1
        self.class_probs = self.value_probabilities(-1)
        
        for i in range(param_size):
            val_dict = self.param_class_values(i)
            self.variable_values.append(val_dict)
            
        self.make_means_variances()
        
    def make_means_variances(self):
        """ 
            Calculates the mean of every parameters value within
            a certain class.
        """
        for param in self.variable_values:
            means = {}
            variances = {}
            
            for class_name in param:
                values = param[class_name]
                means[class_name] = np.mean(values)
                variances[class_name] = np.var(values)
                
            self.variable_means.append(means)
            self.variable_variances.append(variances)
                
    def value_probabilities(self, i):
        """
            Calculates the probability of all values in the column
            and stores them in a dictionary.
        """
        dict = {}
        dataset_size = len(self.training)
        
        for row in self.training:
            val = row[i]
            if val in dict:
                dict[val] += (float(1)/dataset_size)
            else:
                dict[val] = (float(1)/dataset_size)
        return dict
        
    def param_class_values(self, i):
        """
            Takes a row of values for a certain parameters.
            These values are then split by class, so we can later
            calculate the mean and variance per parameter per class.
        """
        dict = {}
        dataset_size = len(self.training)
        
        for row in self.training:
            val = row[i]
            key = row[-1]
            if key in dict:
                values = dict[key]
                values.append(val)
            else:
                dict[key] = [val]
        return dict
    
    """
        Initializing functions
    """
    
    def __init__(self, filename = 'digist123-1.csv', int_format = False):
        """
            The constructor for the GDModel class. When creating the object
            the filename of the csv file should be given as a parameter (as
            a string).
        """
        self.training = self.read(filename)
        
        # Possible values per class per parameter.
        self.variable_values = []
        
        # Variable means and variances per class per parameter.
        self.variable_means = []
        self.variable_variances = []
        
        # Probabilities of classes.
        self.class_probs = {}
        
        # Class probability per test example.
        self.prediction_probabilities = []
        
        self.train()
        
    def read(self, filename):
        """
            Reads every row, splits using the ';' delimiter and converts to
            float before returning an array with all values. Every column except
            the last represents the x values and the last the classification.
        """
        data = []
        
        with open(filename, 'rU') as f:
            reader = csv.reader(f)

            for row_str in reader:
                row = row_str[0].split(';')
                
                for i in range(len(row)):
                    row[i] = float(row[i])
                    
                data.append(row)
            return data
            
    """
        Other functions
    """
    def select_column(self, data, i):
        """
            Returns the ith column of the given array.
        """
        
        col = []
        
        for row in data:
            col.append(data[i])
            
        return col