#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


# the words (features) and authors (labels), already largely processed
# these files should have been created from the previous (Lesson 10) mini-project.
words_file = "../010 - Text Learning/your_word_data.pkl" 
authors_file = "../010 - Text Learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
word_data2 = list()

for i in word_data:
	if "sshacklensf" in i: # second one is "cgermannsf"
		i = i.replace("sshacklensf", "")
	if "cgermannsf" in i:
		i = i.replace("cgermannsf", "")
	if "houectect" in i:
		i = i.replace("houectect", "")
	if "houect" in i:
		i = i.replace("houect", "")
	word_data2.append(i)

authors = pickle.load( open(authors_file, "r") )



# test_size is the percentage of events assigned to the test set (remainder go into training)
# feature matrices changed to dense representations for compatibility with classifier
# functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data2, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


# a classic way to overfit is to use a small number
# of data points and a large number of features
# train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



# your code goes here
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

d_tree = DecisionTreeClassifier()
d_tree.fit(features_train, labels_train)
pred = d_tree.predict(features_test)
print "Accuracy: ", accuracy_score(labels_test, pred)
i_features = d_tree.feature_importances_
count = 0
for f in i_features:
	if(f > 0.2):
		print "f: ", f, "  index: ", count
		break
	count += 1
print vectorizer.get_feature_names()[count]

