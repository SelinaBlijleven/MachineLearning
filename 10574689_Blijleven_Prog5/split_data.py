"""
	Selina Blijleven 	10574689
	
    Programming Assignment 5
    December 2015
"""

import csv
import numpy as np

def main(file_name="digist123-1.csv", train_p = 0.6, validation_p = 0.2, test_p = 0.2):
    """
       Splits the data from a given file into smaller sets given
       three proportions.
    """
    data = read(file_name)
    train, validation, test = divide(data, train_p, validation_p, test_p)
    write(train, "train.csv")
    write(validation, "validation.csv")
    write(test, "test.csv")
    
def divide(data, train_p, validation_p, test_p):
    """
        Shuffles the data and divides according to the given
        proportions.
    """
    full_len = len(data)
    np.random.shuffle(data)
    
    train_len = int(round(full_len * train_p))
    validation_len = int(round(full_len * validation_p))
    
    train_set = data[:train_len]
    validation_set = data[train_len:(train_len + validation_len)]
    test_set = data[(train_len + validation_len):]
    
    return train_set, validation_set, test_set
    
def read(filename):
    """
        Reads every row, splits using the ';' delimiter and converts to
        float before returning an array with all values. Every column except
        the last represents the x values and the last the classification.
    """
    data = []
    
    with open(filename, 'U') as f:
        reader = csv.reader(f)
            
        for row_str in reader:
            row = row_str[0].split(';')
            
            for i in range(len(row)):
                row[i] = float(row[i])
                
            data.append(row)
        return data
        
def write(data, filename):
    """
        Writes the data to the specified file.
    """
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for row in data:
            for i in range(len(row)):
                row[i] = str(int(row[i]))
            
            writer.writerow(row)

if __name__ == "__main__":
    main()