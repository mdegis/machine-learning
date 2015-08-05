#!/usr/bin/python

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

# first element is our labels, any added elements are predictor
# features. Keep this the same for the mini-project, but you'll
# have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# without spliting data part:
clf = DecisionTreeClassifier()
# clf.fit(features,labels)
# print clf.score(features, labels) # 0.989473684211

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
	features, labels, test_size=0.3, random_state=42)
clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test) # 0.724137931034