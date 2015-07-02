# Machine Learning in Python

This repository contains some assignments and exercises of Machine Learning course of Sebastian Thrun and Katie Malone.

Please read the license file. I do **NOT** take any responsibility in case of plagiarism. It's **ONLY** educational purpose.

## Exercises:
### 001:

Decide whether car should go fast or slow depending on bumpiness level of road.

* Plot data with decision line by using Naive Bayes, and save it in 'test.png'
* Calculate accuracy (sklearn build-in)

### 002:

* Same thing in exercises 01 is done again however, this time with SVM and linear kernel.

### 003:

* Same thing in exercises 01 is done again however, this time with Decision Tree.

## Mini-Projects:

Before try to run any of the assignments, please run 'startup.py' in 'tools' directory. This will automatically install e-mail data set and extract it (which is about 423 MB in tgz, 1.4 GB when decompressed).

### Lesson 1: Naive Bayes

* Naive Bayes Classifier is used to identify emails by their authors. 
* Also time performance and accuracy are calculated.

### Lesson 2: SVM

In this mini-project, we’ll tackle the exact same email author ID problem as the Naive Bayes mini-project, but now with an SVM. What we find will help clarify some of the practical differences between the two algorithms. This project also gives us a chance to play around with parameters a lot more than Naive Bayes did, so we will do that too.

Read the comments in the code, for more information.

### Lesson 3: Decision Tree

In this project we'll be tackling the same project that we've done with our last two supervised classification algorithms. We're trying to understand who wrote an email based on the word content of that email. This time we'll be using a decision tree. We'll also dig into the features that we use a little bit more. This'll be a dedicated topic in the latter part of the class. What features give you the most effective, the most accurate supervised classification algorithm?

Read the comments in the code, for more information.

### Lesson 4: AdaBoost (Adaptive Boosting), kNN and Random Forrest

* AdaBoost:

While every learning algorithm will tend to suit some problem types better than others, and will typically have many different parameters and configurations to be adjusted before achieving optimal performance on a dataset, AdaBoost (with decision trees as the weak learners) is often referred to as the best out-of-the-box classifier. When used with decision tree learning, information gathered at each stage of the AdaBoost algorithm about the relative 'hardness' of each training sample is fed into the tree growing algorithm such that later trees tend to focus on harder to classify examples.

A great article about AdaBoost can be found at https://www.cs.princeton.edu/~schapire/papers/explaining-adaboost.pdf

* k Nearest Neighbors: 

Neighbors-based classification is a type of instance-based learning or non-generalizing learning; it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

* Random Forrest:

The Random Forrest (ensemble learning method [like AdaBoost] for classification, regression) method combines Breiman's "bagging" idea and the random selection of features, introduced independently by Ho and Amit and Geman in order to construct a collection of decision trees with controlled variance. The selection of a random subset of features is an example of the random subspace method, which, in Ho's formulation, is a way to implement classification proposed by Eugene Kleinberg.

### Lesson 5: Dataset and Questions

The Enron fraud is a big, messy and totally fascinating story about corporate malfeasance of nearly every imaginable type. The Enron email and financial datasets are also big, messy treasure troves of information, which become much more useful once you know your way around them a bit. We’ve combined the email and finance data into a single dataset, which you’ll explore in this mini-project.

The aggregated Enron email + financial dataset is stored in a dictionary, where each key in the dictionary is a person’s name and the value is a dictionary containing all the features of that person. The email + finance (E+F) data dictionary is stored as a pickle file, which is a handy way to store and load python objects directly. 

### Lesson 6: Regression

In this project, we will use regression to predict financial data for Enron employees and associates. Once we know some financial data about an employee, like their salary, what would you predict for the size of their bonus?

Read the comments in the code, for more information.