"""
	Selina Blijleven 	10574689
	
    Programming Assignment 6
    December 2015
"""
import evaluate_clusters

from kmeans import KMeans
from pylab import *

def write_means(k = 3):
    # Data file
    data = 'digist123-1.csv'
    
    # Max amount of standard deviations for anomaly
    r = 2.0
    
    model = KMeans(data)
    cluster_means, clusters = model.cluster(k, write=True)
    evaluate_clusters.accuracy(clusters, write=True)
    evaluate_clusters.anomalies(clusters, cluster_means, r, write=True)

def compare_accuracies():
    # Data file
    data = 'digist123-1.csv'
    
    # Different k values to test
    no_clusters = range(11)[1:]
    
    # Accuracies per k value
    accuracies = []
    
    # Write the means to a file?
    write = False
    
    # Find k clusters and calculate accuracy
    for k in no_clusters:
        model = KMeans(data)
        _, clusters = model.cluster(k, write)
        accuracies.append(evaluate_clusters.accuracy(clusters, write))
    
    plot_accuracies(no_clusters, accuracies)
    
def plot_accuracies(x, y):
    """
        Plots the accuracy against each value for k.
    """
    plot(x, y)
    xlabel("Number of clusters")
    ylabel("Accuracy in %")
    show()
    
if __name__ == "__main__":
    write_means(4)
    #compare_accuracies()