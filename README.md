# Supervised machine learning algorithms application on dataset with binomial outcome
## Contents
   Abstract
1. [Introduction](#1-introduction)
2. [Exploratory data analysis](#2-exploratory-data-analysis)
3. [Methodology](#3-methodology)
   [SVM](#1-svm)
   [K-nearest neigbors classifier](#2-k-nearest-neigbors-classifier)
   [Logistic Regression](#3-logistic-regression)
   [Decision Tree Classifier](#4-decision-tree-classifier)
   [Random Forest Classifier](#5-random-forest-classifier)
4. [Conclusion](#4-conclusion)  

## Abstract
In the following report I will be discussing several machine learning algorithms and their application on breast cancer dataset. Dependent variable is a diagnosis i.e. I would like to predict whether a patient has a malignant or benign tumor. I will be applying 5 different algorithms: Support vector machines, K nearest neighbor classification, Random forest Classifier, Decision Tree Classifier and Logistic Regression. I will be testing algorithms using K-fold clustering.

## 1. Introduction
Data presented in this report is a breast cancer data (Wisconsin) that is available at UCI ML repository. I selected this data because breast cancer is a very serious and infamous disease that takes a lot of lives every year. I wanted to see which features were the most important in putting diagnosis of benign or malignant cancer. Data has 569 observation with 33 columns-one of them being diagnosis-dependent variable. I dropped two columns called “Unnamed:32” and “ID” because they are redundant. So, for our analysis I will use 30 features to predict diagnosis. There were no missing values found.
## 2. Exploratory data analysis
As a first step of our exploratory data analysis I decide to plot number of malignant versus number of benign diagnosis I have. As you can see from the Figure 1, number of benign diagnosis is 357 and malignant -212. I find data to be balanced and well representative of a daily life data where more people have benign cancer.
![Breast_cancer_project](https://user-images.githubusercontent.com/48894925/76380652-bc2b8880-6329-11ea-831c-e89a83c199b2.jpg)

After I decided to plot histograms of features I have and see how values of the features change depending on the diagnosis. For some of the features, like perimeter_mean, represented in the figure 2, benign and malignant cancer are well differentiated but for some features, like smoothness_mean- measures overlap and I cannot easily differentiate data just by looking at the histogram. I plotted histogram for all the features and realized that just plotting if them is not enough and I need statistical test in order to choose best features.
![2](https://user-images.githubusercontent.com/48894925/76380754-00b72400-632a-11ea-9015-910c3b4773b7.jpg)
I also did a correlation heatmap in order to see if there is multicollinearity. In the figure below you can see just a small part of the correlation heatmap. If two variables are highly correlated, corresponding square will represent Pearson correlation and color from white-highly correlated to dark-not correlated. As I expected, area, perimeter, radius and all corresponding features are perfectly correlated and therefore the reis no need to use all of them But since there are a lot of features manually choosing one by one is time consuming so I will use statistical analysis in order to decide which features to include in the final model.
![3](https://user-images.githubusercontent.com/48894925/76380790-204e4c80-632a-11ea-801c-ba10abf2b424.jpg)
## 3. Methodology
After exploring my dataset, I decided to implement some of the most popular supervised learning algorithms such as Support vector machines, K nearest neighbor classification, Random forest Classifier, Decision Tree Classifier and Logistic Regression.
For all algorithms firstly I train the data and then calculated accuracy of the model on that data. In order to be more objective, I will be testing algorithms using K-fold clustering, to be specific 4-fold clustering. I also defined shuffle=True, so that data is shuffled before divided into 4 folds. After running each algorithm, I also calculate average accuracy of the 4 Cross Validation folds.
  ### 1. SVM
  First algorithm I implemented was Support Vector Machines (SVM). As we know, SVM is a discriminative classifier defined by a separating hyperplane i.e. given dependent variable, the algorithm outputs an optimal hyperplane which categorizes new examples. SVM algorithms use a set of mathematical functions that are defined as the kernel. There are several types of kernels exist but in my analysis I used only linear, radial basis function (RBF) and sigmoid.
Gaussian radial basis function (RBF) is a default Kernel in the python SVM library. It is a general-purpose kernel; used when there is no prior knowledge about the data and uses following formula:


In python, default value for gamma is 1 / (n_features * X.var()). I used this value of gamma for all SVM kernels. My training accuracy for SVM (kernel=rbf) is 100%, however for the 4-fold Cross validation, scores are 67%, 61%, 67%, 65% which makes average accuracy of 65%. Even though this score is low at this point I cannot make any conclusions since I didn’t test other algorithms.
For Linear Kernel the equation for prediction for a new input using the dot product between the input (x) and each support vector (xi) is calculated as follows:


This is an equation that involves calculating the inner products of a new input vector (x) with all support vectors in training data. The coefficients b0 and ai (for each input) must be estimated from the training data by the learning algorithm.
Applying this Kernel gave much better results. Even though accuracy is 96.6%, CV scores are 98%, 93%, 90%, 96% making on average 94.7%. This means that Linear kernel performs much better and therefore is a candidate on best performing algorithm so far.
Lastly, Sigmoid Kernel which is similar to sigmoid function in logistic regression has following formula:

I didn’t have any expectation prior, but the sigmoid kernel came up with the lowest scores. Accuracy score for training data is 62% while CV scores are 62%, 59%, 64%, 64% making on average
  ### 2. K-nearest neigbors classifier
  The KNN algorithm assumes that similar things exist in close proximity i.e. similar things are near to each other. In python, default number of neighbors is 5. I find this number pretty good taking into consideration the fact that we have only two groups: malignant and benign. There are two ways to calculate proximity of the observations to each other. Distance is calculated using Minkowski metric and when p=1it becomes Manhattan distance, when p=2 it is Euclidean distance and for other values of p it calculates Minkowski l_p distance. Figure below show the formulas used I order to calculated specifies distances.
  
![4](https://user-images.githubusercontent.com/48894925/76380846-483db000-632a-11ea-8fd3-09301e7d38d7.jpg)

  For the simplicity in my analysis I included Euclidean, Manhattan and Minkowski p=3 distance.
Euclidean distance is the default method used in the K-nearest Neighbors classifier. For Manhattan, p=1 should be used. Summary of the results for all three types of distances is presented in the table below:


All distance calculation methods give similar and good results results but still lower than SVM linear kernel.
  ### 3. Logistic Regression
  Logistic regression is also called the sigmoid It’s an S-shaped curve that can takes a value and maps it into a value between 0 and 1.

![5](https://user-images.githubusercontent.com/48894925/76380922-7c18d580-632a-11ea-8f0c-2fe444b1eb87.jpg)

where x is the value that you want to transform. Below is a plot of logistic curve.

![6](https://user-images.githubusercontent.com/48894925/76380957-9a7ed100-632a-11ea-88b5-3f37d81139be.jpg)

In order to create less complex model when you have a large number of features in your dataset, some of the Regularization techniques are used to address over-fitting and feature selection. In Python, you can
decide what type of regularization you want to apply to your model. Default is L2 which is Ridge regression, other one is L1- Lasso regression.
Ridge regression:

![7](https://user-images.githubusercontent.com/48894925/76381053-d31eaa80-632a-11ea-8309-7f54f3e65a64.jpg)

Lasso Regression:

![8](https://user-images.githubusercontent.com/48894925/76381054-d3b74100-632a-11ea-90c3-d2ba5506a6ef.jpg)


The key difference between these two is the penalty term.
Logistic regression with ridge penalty
I first applied ridge regression penalty since it was the default one. It gave me the following results:
Training data accuracy is 95.9%, while CV scores are 97.9%, 92.2%,92.9%, 95.7% making average accuracy score 94.7%. Penalty term decreased coefficient terms of all predictors but still all of them were still present in the model. This model gave similarly good results as the Linear kernel of SVM.
Logistic regression with lasso penalty
Prior running code I knew that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. This algorithm gave me following results: Training accuracy score is 95.6% and CV scores are 98.6%, 92.9%, 92.2%, 95.7% making average testing accuracy of 94.8%.
Speaking of coefficients of predictors:

![9](https://user-images.githubusercontent.com/48894925/76381055-d3b74100-632a-11ea-8cc1-1c4e0d969cf0.jpg)

We can see from the matrix above that Lasso penalty removed 18 predictors leaving only 12, which are: radius_mean,texture_mean,perimeter_mean,area_mean, texture_se, perimeter_se, area_se, radius_worst, texture_worst, perimeter_worst, area_worst, concavity_worst.
Since less predictors decrease running time and accuracy score are pretty good at this time, I find this algorithm to be the most optimal one.
  ### 4. Decision Tree Classifier
  Decision tree is one of the most used ML classifiers. Decision tree is a tree-like graph with nodes representing the place where we pick an attribute and ask a question; edges represent the answers the to the question; and the leaves represent the actual output or class label.
When defining Decision Trees, one should first define criterion by which to measure the quality of a split. In python supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. In my analysis I chose default criterion i.e. ‘gini’.
Gini Index for Binary Target variable is
Gini index = 1 — P2 (Target=0) — P2(Target=1)
From the decision tree below you can see gini index at each level, also Value- number of observations per each group and class that shows to which class observations satisfying previous node criteria now belong. I didn’t define max depth, so it goes to the maximum depth possible. At this point one can decide at which point i.e. depth to cut tree.I think I would define depth to be 5 since I can see that at that depth predictions are good. As you can see from the tree, at gini=0.5 i.e. when there is a tie, color of the node is white, but when one class outnumbers other one, then node takes corresponding color. In our case orange for Benign and blue for Malignant. The more intense color, the lower is gini index i.e. more accurate is node prediction.

![10](https://user-images.githubusercontent.com/48894925/76381056-d3b74100-632a-11ea-829a-0a6291e53921.jpg)

As for the previous algorithms, I calculated training accuracy which is 100%-overfitting. CV testing scores are 98%, 90%, 90%, 93% making average of 93.1%.
  ### 5. Random Forest Classifier
  As we know Random forest classifier creates a set of decision trees from randomly selected subset of training set. It then aggregates the votes from different decision trees to decide the final class of the test object. So, most of the things are similar to Decision Tree but now we have bunch of trees. For my analysis I defined number of trees to be 100, for criterion I still use gini index and don’t define maximum
depth. In the figure below you can see Decision tree i.e. 1 of 100 that are present in the Random Forest Classifier.

![11](https://user-images.githubusercontent.com/48894925/76381057-d3b74100-632a-11ea-8b00-f098d46d90d9.jpg)

I calculated training accuracy which is 100%- overfitting too. CV testing scores are 100%, 92%, 90%, 98% making average of 95.2%. i.e. Random Forest becomes one of the best.
## 4. Conclusion
After running 5 most popular Supervised learning algorithms I can conclude that for Breast Cancer dataset Random Forest Classifier, SVM with linear kernel and Logistic Regression with Lasso penalty give the highest accuracy results. I later figured out that running algorithm just for 6 most significant gave for some algorithm even better results and for other the difference was small. For my predictions I would use Top 3 algorithms using both: all predictors and only 6 best predictors, and then compare results.
