#!/usr/bin/python

"""
    starter code for the regression mini-project
    
    loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project)

    draws a little scatterplot of the training/testing data

    you fill in the regression code where indicated

"""    

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
dictionary = pickle.load( open("final_project_dataset_modified.pkl", "r") )

features_list = ["bonus", "salary"]
# features_list = ["bonus", "long_term_incentive"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

reg = LinearRegression()
reg.fit(feature_train, target_train)
print "Slope: ", reg.coef_[0]
print "Intercept: ", reg.intercept_
print "Score of train set: ",reg.score(feature_train, target_train)
print "Score of test set: ",reg.score(feature_test, target_test)

"""
    There are lots of finance features available, some of which might be more 
    powerful than others in terms of predicting a person's bonus. For example, 
    suppose you thought about the data a bit and guess that the 
    "long_term_incentive" feature, which is supposed to reward employees 
    for contributing to the long-term health of the company, might be more 
    closely related to a person's bonus than their salary is.

    A way to confirm that you're right in this hypothesis is to regress the bonus 
    against the long term incentive, and see if the regression score is 
    significantly higher than regressing the bonus against the salary. Perform 
    the regression of bonus against long term incentive
    -- what's the score on the test data?
"""

# draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

# labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

# draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])

"""
    This is a sneak peek of the next lesson, on outlier identification and removal. 
    Go back to a setup where you are using the salary to predict the bonus, and 
    rerun the code to remind yourself of what the data look like. You might notice 
    a few data points that fall outside the main trend, someone who gets a high 
    salary (over a million dollars!) but a relatively small bonus. This is an example 
    of an outlier, and we'll spend lots of time on them in the next lesson.

    A point like this can have a big effect on a regression: if it falls in the training 
    set, it can have a significant effect on the slope/intercept if it falls in the test 
    set, it can make the score much lower than it would otherwise be As things stand 
    right now, this point falls into the test set (and probably hurting the score on 
    our test data as a result). Let's add a little hack to see what happens if it falls 
    in the training set instead.
"""

reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="g") 

print "Second slope: ", reg.coef_[0]
print "Second intercept: ", reg.intercept_

plt.ylabel(features_list[0])
plt.legend()
plt.savefig("regression.png")
plt.show()
