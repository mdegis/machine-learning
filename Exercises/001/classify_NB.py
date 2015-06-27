from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def classify(features_train, labels_train):   
    clf = GaussianNB()
    return clf.fit(features_train, labels_train)
    

def NB_accuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier as a seperate method """

    clf = GaussianNB()
    clf.fit(features_train, labels_train)

    pred = clf.predict(features_test)

    accuracy = accuracy_score(labels_test, pred)
    return accuracy
    
	
