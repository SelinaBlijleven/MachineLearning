"""

	Selina Blijleven 	10574689
	
    
	Programming Assignment 6
    
	December 2015

"""


Question 1:

K-means was succesfully implemented. It is operated by the write_means
function in main.py. The means are re-written every time the program is
 called 
and will be stored in a file called 'means.csv'.



Question 2:

To optimize k an accuracy measure has been defined in
evaluate_clusters.py. This function determines the most frequently 
occurring label in each cluster 
and produces the amount of 'correctly'
labeled examples. This result was then graphed for k 1 through 10, 
multiple times to compensate for random initializing. 
The "elbow", 
where the accuracy still improved greatly lies at k = 4. 
(see 'k_accuracies.png')



Question 3:

Part of this question was already done for question 2, but the
 anomalies are also saved through the write_means function that uses
 a single k. The misclassified 
examples are saved in 'misclassified.txt'
 with the given labels and the cluster labels.



Question 4:

An anomalous example is defined by its euclidian distance. If the
 euclidian distance is outside of r standard deviations it is regarded
 as anomalous. 
The function that implements anomaly detection is called 
anomalies and is found in the file 'evaluate_clusters.py'. The default 
value for r is 2, so an anomaly 
has a bigger distance than twice the
 standard deviation. The anomalies can be found in anomalies.csv.



Question 5:

The anomalies make sense in that it is logical that their euclidian 
distance is bigger. They have a lot of values near 16 and 0 and very
 little values in between, 
which causes them to be near the very ends of
 the parameter space and their euclidian distance to be very high or very
 low. The most anomalous examples are stored 
in 'anomalies.csv'.



Question 6:

The detection algorithm from the Stanford videos uses the chance of the
 anomalous example. To calculate this chance the multivariate Gaussian
 distribution is used. 
This method considers the individual values of
 the features and allows for a threshold to be set on probability. 
The implemented solution, however, 
uses the Euclidian distance. In this 
case the individual variables are not considered, but we consider the
 location in the data space instead. 
This method is obviously less 
precise and therefore worse in detecting anomalies, but is a simpler
 solution that produces logical results.