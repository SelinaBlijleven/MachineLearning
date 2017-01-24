"""
	Selina Blijleven 	10574689
	
    Programming Assignment 6
    December 2015
"""

import csv
import random
import numpy as np

import euclidian
import store_data

class KMeans(object):
    
    """
        Clustering functions.
    """
    
    def cluster(self, k, write=True):
        """
            Generates the initial clusters from generated means,
            then calculates new means and new clusters until the
            clusters do not change anymore.
        """
        self.generateMeans(k)
        self.set_clusters()
        cluster_means = self.iterate()
        
        if write == True:
            store_data.write(cluster_means, "means.csv")
        return cluster_means, self.assignments
        
    def iterate(self):
        """
            Performs iterations over the cluster by re-calculating
            the mean of the cluster and then re-clustering.
        """
        iterations = 0
        
        while self.previous_assignments != self.assignments:
            iterations += 1
            self.set_centroids()
            self.set_clusters()
        return self.centroids
        
    def set_centroids(self):
        """
            Updates all centroids with new means.
        """
        for key in self.assignments.keys():
            examples = self.assignments[key]
            self.update_centroid(key, examples)
            
    def update_centroid(self, key, examples):
        """
            Updates the mean vector for every centroid.
        """
        new = []
        
        for i in range(self.n):
            total = 0
            
            for ex in examples:
                total += ex[i]
            new.append(total / float(len(examples)))
        self.centroids[key] = new
        
    def set_clusters(self):
        """
            Empties the current assignments and uses the minimum
            euclidian distance to assign the examples to the clusters.
        """
        self.previous_assignments = self.assignments
        self.assignments = {}
        
        for example in self.data:
            features = example[:-1]
            min_cluster = 10
            min_distance = 10000
            
            for i in range(len(self.centroids)):
                distance = euclidian.euclidian_distance(features, self.centroids[i])
                if distance < min_distance:
                    min_cluster = i
                    min_distance = distance
            
            self.assign(example, min_cluster)
            
    def assign(self, example, min_cluster):
        """
            Assigns the example to the cluster. This is a help
            function to make sure the list of examples in the 
            cluster is not overwritten.
        """
        if min_cluster in self.assignments.keys():
            self.assignments[min_cluster].append(example)
        else:
            self.assignments[min_cluster] = [example]
                    
    def generateMeans(self, k):
        """
            Uses k random training examples to use as the centroids.
        """
        shuffled_data = self.data
        random.shuffle(shuffled_data)
        self.centroids = shuffled_data[:k]
    
    """
        Init functions
    """
    
    def __init__(self, filename = 'digist123-1.csv'):
        """
            The constructor for the GDModel class. When creating the object
            the filename of the csv file should be given as a parameter (as
            a string).
            
            Data: Complete data from the file.
            m: The amount of examples.
            n: The amount of features/dimensions.
            centroids: The cluster centroids
        """
        self.data = self.read(filename)
        self.m = len(self.data)
        self.n = len(self.data[0]) - 1
        
        self.centroids = []
        self.assignments = {}
    
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