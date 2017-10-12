# Machine Learning in Python

This repository contains some assignments and exercises of Machine Learning course of Sebastian Thrun and Katie Malone.

Please read the license file. I do **NOT** take any responsibility in case of plagiarism. It's **ONLY** educational purpose.

## Mini-Projects:

Before try to run any of the assignments, please run 'startup.py' in 'tools' directory. This will automatically install e-mail data set and extract it (which is about 423 MB in tgz, 1.4 GB when decompressed).

### Lesson 1: Naive Bayes

* Naive Bayes Classifier is used to identify emails by their authors. 
* Also time performance and accuracy are calculated.
* Decide whether car should go fast or slow depending on bumpiness level of road.

![plot](https://github.com/mdegis/machine-learning/raw/master/001%20-%20Naive%20Bayes%20Classifier/exercise/bayes.png)

### Lesson 2: SVM

In this mini-project, we’ll tackle the exact same email author ID problem as the Naive Bayes mini-project, but now with an SVM. What we find will help clarify some of the practical differences between the two algorithms. This project also gives us a chance to play around with parameters a lot more than Naive Bayes did, so we will do that too.

Read the comments in the code, for more information.

![plot](https://github.com/mdegis/machine-learning/raw/master/002%20-%20SVM/exercise/svm_lin.pngg)

### Lesson 3: Decision Tree

In this project we'll be tackling the same project that we've done with our last two supervised classification algorithms. We're trying to understand who wrote an email based on the word content of that email. This time we'll be using a decision tree. We'll also dig into the features that we use a little bit more. This'll be a dedicated topic in the latter part of the class. What features give you the most effective, the most accurate supervised classification algorithm?

Read the comments in the code, for more information.

Overfitted example:
![plot](https://github.com/mdegis/machine-learning/raw/master/003%20-%20Decision%20Tree/exercise/overfitted.png)
Fixed:
![plot](https://github.com/mdegis/machine-learning/raw/master/003%20-%20Decision%20Tree/exercise/dec_tree.png)


### Lesson 4: AdaBoost (Adaptive Boosting), kNN and Random Forrest

* **AdaBoost**:

While every learning algorithm will tend to suit some problem types better than others, and will typically have many different parameters and configurations to be adjusted before achieving optimal performance on a dataset, AdaBoost (with decision trees as the weak learners) is often referred to as the best out-of-the-box classifier. When used with decision tree learning, information gathered at each stage of the AdaBoost algorithm about the relative 'hardness' of each training sample is fed into the tree growing algorithm such that later trees tend to focus on harder to classify examples.

A great article about AdaBoost can be found at https://www.cs.princeton.edu/~schapire/papers/explaining-adaboost.pdf

![plot](https://github.com/mdegis/machine-learning/raw/master/004%20-%20AdaBoost%20%2B%20kNN%20%2B%20%20Random%20Forrest/ada_boost.png)

* **k Nearest Neighbors**: 

Neighbors-based classification is a type of instance-based learning or non-generalizing learning; it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

![plot](https://github.com/mdegis/machine-learning/raw/master/004%20-%20AdaBoost%20%2B%20kNN%20%2B%20%20Random%20Forrest/knn.png)


* **Random Forrest**:

The Random Forrest (ensemble learning method [like AdaBoost] for classification, regression) method combines Breiman's "bagging" idea and the random selection of features, introduced independently by Ho and Amit and Geman in order to construct a collection of decision trees with controlled variance. The selection of a random subset of features is an example of the random subspace method, which, in Ho's formulation, is a way to implement classification proposed by Eugene Kleinberg.

![plot](https://github.com/mdegis/machine-learning/raw/master/004%20-%20AdaBoost%20%2B%20kNN%20%2B%20%20Random%20Forrest/random_forest.png)


### Lesson 5: Dataset and Questions

The Enron fraud is a big, messy and totally fascinating story about corporate malfeasance of nearly every imaginable type. The Enron email and financial datasets are also big, messy treasure troves of information, which become much more useful once you know your way around them a bit. We’ve combined the email and finance data into a single dataset, which you’ll explore in this mini-project.

The aggregated Enron email + financial dataset is stored in a dictionary, where each key in the dictionary is a person’s name and the value is a dictionary containing all the features of that person. The email + finance (E+F) data dictionary is stored as a pickle file, which is a handy way to store and load python objects directly. 

### Lesson 6: Regression

In this project, we will use regression to predict financial data for Enron employees and associates. Once we know some financial data about an employee, like their salary, what would you predict for the size of their bonus?

Read the comments in the code, for more information.

![plot](https://github.com/mdegis/machine-learning/raw/master/006%20-%20Regression/regression.png)


### Lesson 7: Outliers

Having large outliers can have a big effect on your regression result. So in the first part of this mini project, you're going to implement the algorithm that is you take the 10% or so of data points that have the largest residuals, relative to your regression. You remove them, and then you refit the regression, and you see how the result changes.

The second thing we'll do is take a closer at the Enron data. This time with a particular eye towards outliers. You'll find very quickly that there are some data points that fall far outside of the general pattern.

![plot](https://github.com/mdegis/machine-learning/raw/master/007%20-%20Outliers/outlier_fig.png)

### Lesson 8: Unsupervised Learning (K-Means Clustering)

In this project, we'll apply k-means clustering to our Enron financial data. Our final goal, of course, is to identify persons of interest; since we have labeled data, this is not a question that particularly calls for an unsupervised approach like k-means clustering.

Nonetheless, you'll get some hands-on practice with k-means in this project, and play around with feature scaling, which will give you a sneak preview of the next lesson's material.

Great online tool to visualize k-Means Cluster algorithm can be founded at http://www.naftaliharris.com/blog/visualizing-k-means-clustering/

![plot](https://github.com/mdegis/machine-learning/raw/master/008%20-%20K_Means/k_means.png)

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

Read the comments in the code, for more information ([This code](/010 - Text Learning/vectorize_text.py) contains some cool examples).

### Lesson 11: Feature Selection

In one of the earlier videos in this lesson we told you about that there was a word that was effectively serving as a signature on the e-mails and we didn't initially realize it. Now, the mark of a good machine learner doesn't mean that they never make any mistakes or that their features are always perfect. It means that they're on the lookout for ways to check this and to figure out if there is a bug in there that they need to go in and fix.

So in this case it would mean that there's a type of signature word, that we would need to go in and remove in order for us to, to feel like we were being fair in our supervised classification. We will be working on a problem that arose in preparing Chris and Sara’s email for the author identification project; it had to do with a feature that was a little too powerful (effectively acting like a signature, which gives an arguably unfair advantage to an algorithm). You’ll work through that discovery process here.

Read the comments in the [code](/011 - Feature Selection/find_signature.py), for more information.

### Lesson 12: Principal Component Analysis

PCA example by Eigenfaces (face recognition).
![plot](https://github.com/mdegis/machine-learning/raw/master/012%20-%20PCA/eigenfaces.png)

```
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  .. _LFW: http://vis-www.cs.umass.edu/lfw/

  original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html


2015-08-05 22:25:02,927 Loading LFW people faces from /home/mdegis/scikit_learn_data/lfw_home
Total dataset size:
n_samples: 1288
n_features: 1850
n_classes: 7
Extracting the top 150 eigenfaces from 966 faces
done in 4.846s
Projecting the input data on the eigenfaces orthonormal basis
done in 0.372s
Fitting the classifier to the training set
done in 18.366s
Best estimator found by grid search:
SVC(C=1000.0, cache_size=200, class_weight='auto', coef0=0.0, degree=3,
  gamma=0.005, kernel='rbf', max_iter=-1, probability=False,
  random_state=None, shrinking=True, tol=0.001, verbose=False)
Predicting the people names on the testing set
done in 0.061s
                   precision    recall  f1-score   support

     Ariel Sharon       1.00      0.62      0.76        13
     Colin Powell       0.78      0.82      0.80        60
  Donald Rumsfeld       0.87      0.74      0.80        27
    George W Bush       0.82      0.92      0.87       146
Gerhard Schroeder       0.90      0.76      0.83        25
      Hugo Chavez       1.00      0.53      0.70        15
       Tony Blair       0.77      0.75      0.76        36

      avg / total       0.83      0.83      0.82       322

[[  8   2   0   3   0   0   0]
 [  0  49   2   6   0   0   3]
 [  0   0  20   5   0   0   2]
 [  0  11   0 135   0   0   0]
 [  0   0   0   4  19   0   2]
 [  0   0   0   5   1   8   1]
 [  0   1   1   6   1   0  27]]
```

![plot](https://github.com/mdegis/machine-learning/raw/master/012%20-%20PCA/pred.png)

### 013 - Validation

```python
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
```

### Lesson 14: Evaluation Metrics

In the last lesson, you created your first person of interest identifier algorithm, and you set up a training and testing framework so that you could assess the accuracy of that model.

Now that you know much more about evaluation metrics, we're going to have you deploy those evaluation metrics on the framework that you've set up in the last lesson. And so, by the time you get to the final project, the main thing that you'll have to think about will be the features that you want to use, the algorithm that you want to use, and any parameter tunes. You'll already have the training and testing, and the evaluation matrix all set up.
