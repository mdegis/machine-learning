#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess(percentile=1)
clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print "Decision Tree accuracy: %r" % acc

"""
	
	You found in the SVM mini-project that the parameter tune can significantly 
	speed up the training time of a machine learning algorithm. A general rule is 
	that the parameters can tune the complexity of the algorithm, with more 
	complex algorithms generally running more slowly.

	Another way to control the complexity of an algorithm is via the number of 
	features that you use in training/testing. The more features the algorithm 
	has available, the more potential there is for a complex fit. We will explore 
	this in detail in the 'Feature Selection' lesson, but you'll get a sneak preview now.

	What's the number of features in your data?
"""

print "no. of features in your data: %r" % len(features_train[0])
# Change percentile from 10 to 1, and rerun dt_author_id.py. 
# What's the number of features now? 379, before: 3785
"""

	Would a large value for percentile lead to a more complex or less complex decision tree, 
	all other things being equal? 

	* More complex

	Accuracy percentile = 1  : 0.96587030716723554
	Accuracy percentile = 10 : 0.97838452787258245
"""