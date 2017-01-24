"""
	Selina Blijleven 	10574689
	
    Programming Assignment 6
    December 2015
"""

import csv
import os

def write(data, filename):
    """
        Writes the data to the specified file.
    """
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for row in data:
            for i in range(len(row)):
                row[i] = str(int(round(row[i])))
            
            writer.writerow(row)
            
def write_misclassified(data, filename):
    """
        Writes the misclassified examples to a file, along with
        their labels and the class they were mislabeled as.
    """
    f = file(filename, "w")

    for row in data:
        example = row[0]
        cluster_label = row[1]
        example_label = row[0][-1]
        
        f.write(str(example) + "\n")
        f.write("This example was classified as: "  + str(example_label) + "." + "\n")
        f.write("The most frequent label in the cluster was: " + str(cluster_label) + "." + "\n")