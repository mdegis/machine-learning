# Machine Learning in Python

This repository contains some assignments and exercises of Machine Learning course of Sebastian Thrun and Katie Malone.

Please read the license file. I do **NOT** take any responsibility in case of plagiarism. It's **ONLY** educational purpose.

## Mini-Projects:

Before try to run any of the assignments, please run 'startup.py' in 'tools' directory. This will automatically install e-mail data set and extract it (which is about 423 MB in tgz, 1.4 GB when decompressed).

### Lesson 1: Naive Bayes

* Naive Bayes Classifier is used to identify emails by their authors. 
* Also time performance and accuracy are calculated.
* Decide whether car should go fast or slow depending on bumpiness level of road.

![plot](/001 - Naive Bayes Classifier/exercise/bayes.png)

### Lesson 2: SVM

In this mini-project, we’ll tackle the exact same email author ID problem as the Naive Bayes mini-project, but now with an SVM. What we find will help clarify some of the practical differences between the two algorithms. This project also gives us a chance to play around with parameters a lot more than Naive Bayes did, so we will do that too.

Read the comments in the code, for more information.

![plot](/002 - SVM/exercise/svm_lin.png)

### Lesson 3: Decision Tree

In this project we'll be tackling the same project that we've done with our last two supervised classification algorithms. We're trying to understand who wrote an email based on the word content of that email. This time we'll be using a decision tree. We'll also dig into the features that we use a little bit more. This'll be a dedicated topic in the latter part of the class. What features give you the most effective, the most accurate supervised classification algorithm?

Read the comments in the code, for more information.

Overfitted example:
![plot](/003 - Decision Tree/exercise/overfitted.png)
Fixed:
![plot](/003 - Decision Tree/exercise/dec_tree.png)


### Lesson 4: AdaBoost (Adaptive Boosting), kNN and Random Forrest

* **AdaBoost**:

While every learning algorithm will tend to suit some problem types better than others, and will typically have many different parameters and configurations to be adjusted before achieving optimal performance on a dataset, AdaBoost (with decision trees as the weak learners) is often referred to as the best out-of-the-box classifier. When used with decision tree learning, information gathered at each stage of the AdaBoost algorithm about the relative 'hardness' of each training sample is fed into the tree growing algorithm such that later trees tend to focus on harder to classify examples.

A great article about AdaBoost can be found at https://www.cs.princeton.edu/~schapire/papers/explaining-adaboost.pdf

![plot](/004 - AdaBoost + kNN +  Random Forrest/ada_boost.png)

* **k Nearest Neighbors**: 

Neighbors-based classification is a type of instance-based learning or non-generalizing learning; it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

![plot](/004 - AdaBoost + kNN +  Random Forrest/knn.png)


* **Random Forrest**:

The Random Forrest (ensemble learning method [like AdaBoost] for classification, regression) method combines Breiman's "bagging" idea and the random selection of features, introduced independently by Ho and Amit and Geman in order to construct a collection of decision trees with controlled variance. The selection of a random subset of features is an example of the random subspace method, which, in Ho's formulation, is a way to implement classification proposed by Eugene Kleinberg.

![plot](/004 - AdaBoost + kNN +  Random Forrest/random_forest.png)


### Lesson 5: Dataset and Questions

The Enron fraud is a big, messy and totally fascinating story about corporate malfeasance of nearly every imaginable type. The Enron email and financial datasets are also big, messy treasure troves of information, which become much more useful once you know your way around them a bit. We’ve combined the email and finance data into a single dataset, which you’ll explore in this mini-project.

The aggregated Enron email + financial dataset is stored in a dictionary, where each key in the dictionary is a person’s name and the value is a dictionary containing all the features of that person. The email + finance (E+F) data dictionary is stored as a pickle file, which is a handy way to store and load python objects directly. 

### Lesson 6: Regression

In this project, we will use regression to predict financial data for Enron employees and associates. Once we know some financial data about an employee, like their salary, what would you predict for the size of their bonus?

Read the comments in the code, for more information.

![plot](/006 - Regression/regression.png)


### Lesson 7: Outliers

Having large outliers can have a big effect on your regression result. So in the first part of this mini project, you're going to implement the algorithm that is you take the 10% or so of data points that have the largest residuals, relative to your regression. You remove them, and then you refit the regression, and you see how the result changes.

The second thing we'll do is take a closer at the Enron data. This time with a particular eye towards outliers. You'll find very quickly that there are some data points that fall far outside of the general pattern.

![plot](/007 - Outliers/outlier_fig.png)

### Lesson 8: Unsupervised Learning (K-Means Clustering)

In this project, we'll apply k-means clustering to our Enron financial data. Our final goal, of course, is to identify persons of interest; since we have labeled data, this is not a question that particularly calls for an unsupervised approach like k-means clustering.

Nonetheless, you'll get some hands-on practice with k-means in this project, and play around with feature scaling, which will give you a sneak preview of the next lesson's material.

Great online tool to visualize k-Means Cluster algorithm can be founded at http://www.naftaliharris.com/blog/visualizing-k-means-clustering/

![plot](/008 - K_Means/k_means.png)

### Lesson 9: Feature Scaling

In the mini-project on K-means clustering, you clustered the data points. And then at the end, we sort of gestured towards feature scaling as something that could change the output of that clustering algorithm. In this mini-project, you'll actually deploy the feature scaling yourself. So you'll take the code from the K-means clustering algorithm and add in the feature scaling and then in doing so, you'll be recreating the steps that we took to make those new clusters.

```python
salary = []
for i in data_dict:
    if (data_dict[i][feature_1]=='NaN'):
        salary.append(0.0)
        # pass
    else:    
        salary.append(float(data_dict[i][feature_1]))
ma= max(salary)        
mi=min(salary)
print "salary maximum: ", ma, " minimum: ", mi
# maximum:  1111258.0  minimum:  477.0 comment out line 121 to get rid of zeroes.
print float(200000-mi)/(ma-mi)

salary_ = numpy.array(salary)
salary_ = salary_.astype(float)
scaler = MinMaxScaler()
rescaled_salary = scaler.fit_transform(salary_)
print rescaled_salary
```

### Lesson 10: Text Learning

In the beginning of this class, you identified emails by their authors using a number of supervised classification algorithms. In those projects, we handled the preprocessing for you, transforming the input emails into a TfIdf so they could be fed into the algorithms. Now you will construct your own version of that preprocessing step, so that you are going directly from raw data to processed features.

You will be given two text files: one contains the locations of all the emails from Sara, the other has emails from Chris. You will also have access to the parseOutText() function, which accepts an opened email as an argument and returns a string containing all the (stemmed) words in the email.

Read the comments in the code, for more information ([This code](/010 - Text Learning/vectorize_text.py) contains some cool examples for python).