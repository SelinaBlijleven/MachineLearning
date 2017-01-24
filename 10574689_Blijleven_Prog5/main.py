"""
	Selina Blijleven 	10574689
	
    Programming Assignment 5
    December 2015
"""

from bayes import GaussianNaiveBayes

def main():
    train_file = 'digist123-1.csv'
    test_file = 'digist123-2.csv'
    
    """ 1b
        To find the 10 most likely misclassified examples in the
        test file the probability of the prediction class was
        divided by the probability of the actual classification.
        This list was sorted by value. From the last ten values eight
        have value two, while the value one was predicted. From all
        24 inaccurate predictions 12 are twos predicted as ones.
        
        When testing the problem sometimes occurred that the variance
        became 0 (which caused a problem when dividing by zero). This
        problem occurred because the value for this parameter was
        always the same for that class. This meant that the chance
        the variable had a certain value x for class c was always one.
        
        However, when this happens with multiple variables the
        probability for this class might be a lot higher than the
        other classes. It might be wise to drop the variable for any
        class where this happens and then implement another measure
        to penalize the probability such that the amount of used
        variables is taken into account.
    """
    
    print "Training Gaussian Naive Bayes model..."
    model = GaussianNaiveBayes(train_file)
    print "Model completed. \n"
    print "Testing model..."
    acc = model.test_model(test_file, misclassified=True)
    print "Testing completed. Accuracy is " + str(acc) + "%. \n"

if __name__ == "__main__":
    main()