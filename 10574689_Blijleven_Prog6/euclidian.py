"""
	Selina Blijleven 	10574689
	
    Programming Assignment 6
    December 2015
"""

import numpy as np

def euclidian_distance(example, neighbour):
    """
        Calculates the euclidian distance between two vectors.
    """
    distance = 0
    
    for i in range(len(example)):
        distance += (example[i] - neighbour[i])**2
        
    return np.sqrt(distance)
    
def euclidian_distance2(example, neighbour):
    """
        Anomaly detection version
        Calculates the euclidian distance between two vectors.
        Writing to a file messed with the means such that they could
        only be returned as strings. This version has an extra cast to
        integer function to patch this problem.
    """
    distance = 0
    
    for i in range(len(example)):
        distance += (example[i] - int(neighbour[i]))**2
        
    return np.sqrt(distance)