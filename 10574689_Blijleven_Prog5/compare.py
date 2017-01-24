"""
	Selina Blijleven 	10574689
	
    Programming Assignment 5
    December 2015
"""

import split_data
from bayes import GaussianNaiveBayes
from k_nearest import KModel

def main():
    data_file = "digist123-1.csv"
    split_data.main(data_file, 0.6, 0.2, 0.2)
    
    train_file = "train.csv"
    validation_file = "validation.csv"
    test_file = "test.csv"
    
    k_vals = [4, 6, 8, 10, 12]
    
    """
        Gaussian Naive Bayes
    """
    # Training
    print "Training Gaussian Naive Bayes model..."
    gnb_model = GaussianNaiveBayes(train_file)
    print "Model completed. \n"
    
    # No parameters for cross validation :(
    
    # Testing
    print "Testing model..."
    gnb_acc = gnb_model.test_model(test_file, misclassified=False)
    print "Testing completed. Accuracy is " + str(gnb_acc) + "%. \n"
    
    """
        K-nearest
    """
    # Training
    print "Training K-nearest model..."
    k_model = KModel(train_file)
    print "Model completed. \n"
    
    # Cross-validation
    print "Cross-validating model..."
    accuracies = []
    
    for k in k_vals:
        cross_acc = k_model.predict(test_file, k, weighted = False, param_weighted = False)
        accuracies.append([cross_acc, k])
    sorted(accuracies)
    print "Cross-validating completed. \n"
    
    # Testing
    print "Testing model..."
    k_acc = k_model.predict(test_file, accuracies[-1][1], weighted = False, param_weighted = False)
    print "Testing completed. Accuracy is " + str(k_acc) + "%. \n"

if __name__ == "__main__":
    main()