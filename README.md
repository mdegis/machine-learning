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