#!/usr/bin/python

""" The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """


from prep_terrain_data import makeTerrainData
from sklearn.metrics import accuracy_score
from class_vis import prettyPicture, output_image
from classify_NB import classify, NB_accuracy
import numpy as np
import pylab as pl
from PIL import Image

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

clf = classify(features_train, labels_train)

### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test)
Image.open('test.png').show()

# JSON object to read data:
# output_image("test.png", "png", open("test.png", "rb").read())

pred = clf.predict(features_test)

print "Naive Bayes accuracy: %r" % accuracy_score(labels_test, pred)




