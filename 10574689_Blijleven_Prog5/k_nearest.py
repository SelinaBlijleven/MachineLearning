"""
    Joram Wessels 		10631542
	Selina Blijleven 	10574689
	
    Programming Assignment 4
    November 2015
"""
import csv
import numpy as np

class KModel(object):
    
    """
        Main functions
    """
    def predict(self, filename = 'digist123-2.csv', k = 3, weighted = False, param_weighted = False):
        """
            Predicts a value for each row based on k neighbours.
            The predictions are then evaluated against their real values
            and the accuracy is returned.
        """
        
        self.test = self.read(filename)
        
        if param_weighted == False:
            predictions = self.plain_predict(k, weighted)
        else:
            predictions = self.variable_predict(k, weighted)
            
        accuracy = self.evaluate(predictions)
        return accuracy
        
    def plain_predict(self, k, weighted):
        """ 
            Predicts without weighting the variables.
        """
        predictions = []
        
        for row in self.test:
                parameters = row[:-1]
                predicted = self.h(parameters, k, weighted)
                predictions.append(predicted)
        return predictions
        
    def variable_predict(self, k, weighted):
        """
            Predicts with weighted variables.
        """
        predictions = []
        weights = self.getWeights(k, weighted)
        
        for row in self.test:
                parameters = row[:-1]
                predicted = self.h_varweight(parameters, k, weighted, weights)
                predictions.append(predicted)
        return predictions
        
    def getWeights(self, k, weighted):
        """
            Determines weights by looking at the effectiveness of the variables.
        """
        predictions = self.getPredictions(k, weighted)
        accuracies = self.getAccuracies(predictions)
        normalized = self.normalize(accuracies)
        return normalized
       
    def normalize(self, values):
        """
            Normalizes the accuracies to use as weights.
        """
        total = sum(values)
        normalized = []
        
        for val in values:
            normalized.append(val / float(total))
        return normalized
       
    def getAccuracies(self, predictions):
        """
            Calculates the accuracy of every individual parameter.
        """
        accs = []
        
        for parameter in predictions:
            accuracy = self.evaluate(parameter)
            accs.append(accuracy)
        return accs
       
    def getPredictions(self, k, weighted):
        """
            Predicts a value for all individual parameters and returns an array
            with the predictions, ordered by parameter.
        """
        all_predictions = []
        
        for i in range(len(self.test[0])):
            param_prediction = []
            
            for row in self.test:
                parameter = row[i]
                prediction = self.h([parameter], k, weighted)
                param_prediction.append(prediction)
            
            all_predictions.append(param_prediction)
        return all_predictions
       
    def evaluate(self, predictions):
        """
            Compares given answers to real answers and returns the 
            accuracy (percentage)
        """
        
        correct = 0
        
        for i in range(len(predictions)):
            if predictions[i] == self.test[i][-1]:
                correct += 1
                
        return float(correct) / len(predictions) * 100
    
    """
        Classifying functions
    """
    def h(self, parameters, k, weighted):
        """
            Returns the hypothesis based on k-nearest neighbours.
            If the parameter weighted is set to true the neighbours
            will be weighted by distance.
        """
        neighbours = self.getNeighbours(parameters, k)
        
        if weighted == False:
            return self.unweightedH(neighbours, k)
                
        if weighted == True:
            return self.weightedH(neighbours, k)
            
    def h_varweight(self, parameters, k, weighted, weights):
        """
            Calculates the weights for each neighbour (these add up to 1)
            and uses these to predict a hypothesis.
        """
        neighbours = self.getNeighbours(parameters, k)
        nb_weights = self.getNeighbourWeights(neighbours, weights)
        
        total = 0
        
        for i in range(len(neighbours)):
            total += neighbours[i][1] *  nb_weights[i]
        return round(total)
        
    def getNeighbourWeights(self, neighbours, weights):
        """
            Calculates all the parameters with their weights
            and divides them by the total to find a weight for
            this neighbour.
        """
        nb_values = []
        
        for i in range(len(neighbours)):
            x = neighbours[i][2]
            nb_value = 0
            
            for j in range(len(x)):
                nb_value += x[j] * weights[j]
            
            nb_values.append(nb_value)
            
        total = sum(nb_values)
        
        for i in range(len(nb_values)):
            nb_values[i] = nb_values[i] / total
        return nb_values
            
    def unweightedH(self, neighbours, k):
        """
            Returns the class of the neighbours weighted by the amount
            of neighbours.
        """
        sum = 0
        
        for nb in neighbours:
                sum += nb[1]
        return round(sum/k)
        
    def weightedH(self, neighbours, k):
        """
            Returns the hypothesis of the nearest neigbours weighted
            by their distance from the test point. (Nearer neighbours
            are considered more important in this case)
        """
        td = 0
        sum = 0
        
        for nb in neighbours:
            td += nb[0]
        
        for nb in neighbours:
            d = nb[0]
            value = nb[1]
            wd = d / td
            sum += wd * float(value)
        return round(sum)
        
    def getNeighbours(self, example, k):
        """
            Sorts the neighbours by euclidian distance and returns the 
            first k.
        """
        distances = []
        
        for row in self.training:
            distance = self.euclidianDistance(example, row[:-1])
            distances.append([distance, row[-1], row[:-1]])
            
        nearest = sorted(distances)
        return nearest[:k]
        
    def euclidianDistance(self, example, neighbour):
        """
            Calculates the euclidian distance between two vectors.
        """
        distance = 0
        
        for i in range(len(example)):
            distance += (example[i] - neighbour[i])**2
            
        return np.sqrt(distance)
    
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