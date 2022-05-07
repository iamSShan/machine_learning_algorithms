## Random Forest

### Introduction

* Random forest, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction.
(In statistics and machine learning, ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.) 

* Random forest is usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.
* Bagging is also called as bootstrap aggregation.
* Let's say if we have a dataset D then we will sample this dataset into n samples and we have n models(decision trees), so from this dataset we will pick some sample rows and sample of features in each sample. Selecting sample of rows is done using row sampling with replacement. And sample of features is called as feature sampling. Each decision tree is trained with different sample data. As we are using sample with replacement, then some of records can be common in different datasets. Similar is the case  with feature sampling, feature may get repeated in datasets sample.

* Sampling with replacement is when a unit selected at random from the population is returned to the population and then a second element is selected at random. Whenever a unit is selected, the population contains all the same units, so a unit may be selected more than once. There is no change at all in the size of the population at any stage. This is only a theoretical concept, and in practical situations the sample is not selected by using this selection method. Suppose a population size N=5 and sample size n=2, and sampling is done with replacement. Out of 5 elements, the first element can be selected in 5 ways. The selected unit is returned to the main lot and now the second unit can also be selected in 5 ways.

* Now when to predict, we give that test sample to every trained model and (in case of binary classification) we get final output by majority vote.

* One big advantage of random forest is that it can be used for both classification and regression problem.

* The low correlation between models is the key. Uncorrelated models can produce ensemble predictions that are more accurate than any of the individual predictions. The reason for this is that the trees protect each other from their individual errors(as long as they don’t constantly all err in the same direction). While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction.


* When we are using decision trees. It has two properties:
	* Low bias: If we are creating decision tree to its complete depth, then it will be properly trained for our training dataset. 
	* High variance: If we give new test data to these decision tree (created till it's complete depth) is more prone to give high error.
	* Using above two points we can see we are getting overfitting condition.
* Now in random forest, we have multiple decision trees, each decision tree will have high variance, but when we combine all decision tree wrt majority vote then high variance is converted into low variance because when we are using row and feature sampling and giving records to decision trees, these trees tend to become expert wrt specific rows and dataset they have as input. Also we are taking majority vote, we are not dependent on a single decision tree. If we would have not used sampling with replacement, then variance could have been high.
* So random forest gives: low bias and low variance.

* Even if we change some rows in existing dataset, like if we had 1000 rows we updated 200 rows, now will this change will impact random forest? Answer is no because these changed records will be also splitted between n decision trees and it will be trained properly. So this data change will not make that much impact on random forest output or accuracy.

* Suppose if it was a regression problem, then for a test data we would have got a continuous value from each model, then we will either take mean or median of the output. Mean or median is decided based on output which each decision tree is giving. Usually sklearn gives output using mean.

### Important points to rememeber about random forest

* Why is it called a "random" forest? Answer: "forest" because there are several trees, "random" because each tree is only trained on a random subset of samples drawn from the training set (with repetition) and possibly a random subset of features. The "random" part is needed because otherwise, the trees would be so similar that there would be no advantage in having a "forest". Randomness makes sure that the trees are uncorrelated.

* When using sklearn library, in the case of random forest classifier has a parameter called **criterion** where either 'gini' or 'entropy' can be written. But in random forest regressor, the value of **criterion** parameter is 'mse'.

* No feature scaling is required here.

* It is robust to outliers because decision trees are also robust to outiers. Also in random forest, we use row and column sampling, so because of this our outlier won't affect our model much.

* Some metrics to be considered here:
	* for classification model: Confusion Matrix, Precision, Recall, F1 score
	* for regression mode: R2, Adjusted R2, MSE, RMSE, MAE

* Cost Function for random forest: Gini impurity or Entropy, in case of regression it is MSE

### Hyperparameters in random forest

* Some important hyperparameters in random forest:
	* To increase predictive power:
		* n_estimators: It is just the number of trees the algorithm builds before taking the maximum voting or taking the averages of predictions. In general, a higher number of trees increases the performance and makes the predictions more stable, but it also slows down the computation.
		* max_features: It is the maximum number of features random forest considers to split a node. Sklearn provides several options here.
		* min_sample_leaf: This determines the minimum number of leafs required to split an internal node.
	* To increase model speed:
		* n_jobs: This hyperparameter tells the engine how many processors it is allowed to use. If it has a value of one, it can only use one processor. A value of “-1” means that there is no limit.
		* random_state: This hyperparameter makes the model’s output replicable. The model will always produce the same results when it has a definite value of random_state and if it has been given the same hyperparameters and the same training data.
		* oob_score(also called oob sampling): It is a random forest cross-validation method. In this sampling, about one-third of the data is not used to train the model and can be used to evaluate its performance. These samples are called the out-of-bag samples. It's very similar to the leave-one-out-cross-validation method, but almost no additional computational burden goes along with it.


### Advantages and Disadvantages of random forest:
* Advantages:
	* Doesn't overfit
	* Ensemble learning algorithms are favourite algorithm for Kaggle competition
	* Less Parameter Tuning required
	* Decision Tree can handle both continuous and categorical variables.
	* No feature scaling (standardization and normalization) required in case of Random Forest as it uses Decision Tree internally.
	* Suitable for any kind of ML problems


* Disadvantages:
	* Biased with features having many categories.
	* Biased in multiclass classification problems towards more frequent classes.


## Can Random Forest Overfit in any case?
* Let's say if after the first split, there are 2 nodes here and we have set the maximum terminal nodes as 2. Hence, the tree will terminate here and will not grow further. This is how setting the maximum terminal nodes or max_leaf_nodes can help us in preventing overfitting. 

* But, it can also overfit. this could simply be due to small sample sizes, but more often, the issue is non-random sampling. All statistical modelling methods are based on theory that assumes the data to be a randomly drawn sample from the population it represents. In practice though, models are often developed using data that does not meet that requirement.	
* For example, if data has some noise in it, two very deep trees will almost surely overfit. You will be (over)fitting the noise.