#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# clf = SVC(kernel="linear")
clf = SVC(kernel="rbf", C=10000)
t0 = time()
"""

	One way to speed up an algorithm is to train it on a smaller training 
	dataset. The tradeoff is that the accuracy almost always goes down when 
	you do this. Let's explore this more concretely: add in the following two 
	lines immediately before training your classifier. 
	
	original (linear):
	training time: 188.996 s            
	predict time: 20.275 s                                                          
	SVM accuracy: 0.98407281001137659   

	These lines effectively slice the training dataset down to 1 percent of its 
	original size, tossing out 99 percent of the training data. 

	Sliced (linear):
	training time: 0.09 s
	predict time: 0.961 s
	accuracy: 0.88452787258248011

	If speed is a major consideration (and for many real-time machine learning 
	applications, it certainly is) then you may want to sacrifice a bit of 
	accuracy if it means you can train/predict faster.

	Different Kernel:
	clf = SVC(kernel="rbf", C=10000)
	Also, C is very effective in this assignment, try to change it and see.

		Sliced data set:
		training time: 0.098 s
		predict time: 0.833 s
		accuracy: 0.89249146757679176

		Full sized data set:
		training time: 118.729 s
		predict time: 13.075 s
		accuracy: 0.99089874857792948 #FTW

"""
# comment out those two lines if you want to see original one
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

# originally: 0.98407281001137659 acc... FTW but it takes time
print "SVM accuracy: %r" % accuracy_score(pred, labels_test)

"""
	What class does your SVM (0 or 1, corresponding to Sara and Chris respectively) 
	predict for element 10 of the test set? The 26th? The 50th? 
	(Use the RBF kernel, C=10000, and 1% of the training set. Normally you'd get 
	the best results using the full training set, but we found that using 1% sped up 
	the computation considerably and did not change our results--so feel free to use 
	that shortcut here.)

"""
print "10th: %r, 26th: %r, 50th: %r" % (pred[10], pred[26], pred[50])

# There are over 1700 test events, how many are predicted to be in the "Chris" (1) class?
print "No. of predicted to be in the 'Chris'(1): %r" % sum(pred)

"""
	Hopefully it’s becoming clearer what they told us about the Naive Bayes -- is 
	great for text -- it’s faster and generally gives better performance than an SVM 
	for this particular problem. Of course, there are plenty of other problems where 
	an SVM might work better. Knowing which one to try when you’re tackling a problem 
	for the first time is part of the art and science of machine learning. In addition 
	to picking your algorithm, depending on which one you try, there are parameter 
	tunes to worry about as well, and the possibility of overfitting (especially if 
	you don’t have lots of training data).

	Our general suggestion is to try a few different algorithms for each problem. 
	Tuning the parameters can be a lot of work, but just sit tight for now--toward 
	the end of the class we will introduce you to GridCV, a great sklearn tool that 
	can find an optimal parameter tune almost automatically.

"""