from PIL import Image
import sys
sys.path.append("../001/")
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()


clf = DecisionTreeClassifier(min_samples_split=50)

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

clf.fit(features_train, labels_train)

#### store your predictions in a list named pred

pred = clf.predict(features_test)

prettyPicture(clf, features_test, labels_test)
Image.open('test.png').show()

acc = accuracy_score(pred, labels_test)
print "Decision Tree accuracy: %r" % acc

"""
	clf = DecisionTreeClassifier(min_samples_split=2)
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	acc_min_samples_split_2 = accuracy_score(pred, labels_test)

	clf = DecisionTreeClassifier(min_samples_split=50)
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	acc_min_samples_split_50 = accuracy_score(pred, labels_test)


	def submit_accuracies():
	  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
	          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}

	submit_accuracies()

"""