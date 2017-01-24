"""
	Selina Blijleven 	10574689
	
    Programming Assignment 6
    December 2015
"""

import store_data
import euclidian
import numpy as np

def accuracy(clusters, write = True):
    """
        Calculates the proportion of examples with the same label
        as the most frequent label in the cluster.
    """
    misclassified = []
    correct = 0
    total = 0
    
    for cluster in clusters.keys():
        examples = clusters[cluster]
        counts = get_frequencies(examples)
        cluster_label = get_max(counts)
        
        for ex in examples:
            label = ex[-1]
            if label != cluster_label:
                misclassified.append([ex, cluster_label])
            else:
                correct += 1
            total += 1
                
    if write == True:
        store_data.write_misclassified(misclassified, "misclassified.txt")
    return correct / float(total) * 100
    
def anomalies(clusters, means, r = 2.0, write = True):
    """
        Detects the anomalies based on the Gaussian. Euclidian
        distances farther then r times the standard deviation are
        considered abnormal.
    """
    anomalies = []
    
    for cluster in clusters.keys():
        examples = clusters[cluster]
        distances = get_distances(examples, means[cluster])
        std_dis = np.std(distances)
        mean_dis = np.mean(distances)
        
        for i in range(len(distances)):
            if np.absolute(distances[i] - mean_dis) > std_dis * r:
                anomalies.append(examples[i])
                
    store_data.write(anomalies, "anomalies.csv")
        
def get_distances(examples, mean):
    """
        Returns a list of euclidian distances.
    """
    distances = []
    
    for ex in examples:
        d = euclidian.euclidian_distance2(ex[:-1], mean)
        distances.append(d)
        
    return distances
        
def get_frequencies(examples):
    """
        Returns a dictionary with the frequency of all labels.
    """
    freq = {}
    
    for ex in examples:
        label = ex[-1]
        
        if label in freq.keys():
            freq[label] += 1
        else:
            freq[label] = 1
            
    return freq
    
def get_max(counts):
    """
        Returns the label from a dictionary with the highest count.
    """
    max = 0
    max_label = 0
    
    for label in counts.keys():
        if counts[label] > max:
            max = counts[label]
            max_label = label
    return max_label