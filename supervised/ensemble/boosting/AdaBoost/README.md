## AdaBoost(Adaptive Boosting):

* Adaboost combines multiple weak learners into a single strong learner. 

* Here let's say we have three independent features(f1, f2, f3) and one dependent feature. Then in first step all records are given some weight, so we can create a new column called sample_weight. To assign sample_weight we apply formula 1/n, where is n is number of records. Let's say here n was 7, then for each row, `sample_weight` will have 1/7 as value.
* In the step two, we create our base learner(i.e. first model). In AdaBoost, all base learners are decision trees.
* Here decision tree are created of only one depth(root node is at depth 0, so depth 1 means root and its children). These decision trees are called stumps.
* For each independent features we create stump(i.e for 3 features we create three decision trees).
* Now point is how to select first base learning model, which of these three is to be selected, we calculate entropy(or gini whatever we want to use) for each stump, and stump whose entropy is least, we select them as our first base learning model.

* Let's say `f1` stump had least entropy and it is selected as first base learning model. Let's say, this whole scenario is of binary classification and if this first selected model has correctly classified four records and incorrectly classified one record. Then for this incorrect classification we find out total error. Total error is calculated by summing up all the incorrect sample weights. Here we have just one incorrect record. So our total error will be 1/7

* In step 3, we calculate performance of stump using formula = 1/2 * log(to the base 3) ( (1-TE) / TE); TE is Total Error. In our case if we use this formula: 1/2 * log(1-1/7 / 1/7 ) => 1/2 * log(6) => 0.8958

* Now what we are trying to do? As we know in boosting we have to pass incorrectly classified records to next base learning model, so here we are just trying to increase the sample weight of all incorrectly classified records, and also we have to decrease the weight of correctly classified records.

* In step 4, we will update the sample weights. To update the weights we have two formula:
	* First we have to update the weights of incorrectly classified records using formula:
		* New sample weight = Current Weight * e^PerformanceSay;  (where PerformanceSay is our performance which we have calculated above.
		* So in this case => 1/7 * e^0.895 => 0.349 (we can see it has increase from 1/7(i.e 0.142) to 0.349)
	* Then we have to update correctly classified points, we will make minor change in formula:
		* New sample weight = Current Weight * e^-PerformanceSay
		* In our example we get: New sample weight = 1/7 * e^-0895 => 0.05
		* So here every correctly classfied point will be made 0.05

	* All updated weights can be stored in a new column called `upadted_weight`.		
	* But if we sum all updated weights we won't get 1, but in case of sample weights sum of all was 1. So here we just divide each entry in updated weights column by total sum of updated_weight column.
	* If entries in updated_weight column were: 0.05, 0.05, 0.349, 0.05, 0.05, 0.05, 0.05, if we sum it we get 0.649, so we divide each entry with 0.649 so updated_weight column becomes = 0.07, 0.07, 0.53, 0.07, 0.07, 0.07, 0.07
	* We can store these weights in column are `normalized_weights`

* In next step, we will create a new dataset, we will mostly select wrong records based on these normalized values to train the model. We will first bucketize the normalized weights, our buckets will be 0 to 0.07, 0.07 to 0.14, 0.14 to 0.60, 0.60 to 0.67, 0.67 to 0.74....and so on (basically we adding 0.07 at each step; 0.60 came from adding 0.07 to 0.53 weight)
* Now our algo will run 8 iteration(as 7 are total records) to select different different records from old dataset. Let' say in first iteration a random value of 0.43 is selected then we will check in which bucket this random value fall, we can see it falls in the incorreclty classified record bucket, then wrong record will be added to the new dataset. If again random value comes 0.31, then again incorrectly classified record bucket will be selected. Similarly it will be going on. Note that, a single record can be inserted in new data more than once.

* Now again based on this new dataset we will create new stump, we will use same methodology as discussed in above points. But in the second base learner model we won't be using `sample_weights`, we will use `normalized_weights`.

* Now let's say there were 3 stumps, these got trained, now for test data, we will pass it to all three stumps and let's say first stump outputs 1, second gives 0, third gives 1. Then we can select on basis of majoirty vote. So here we can 1 will be given as output.

* So here we can say that, in AdaBoost we are combining weak learners(multiple base learning models) and we are making it as strong learner.

* AdaBoost is adaptive in the sense that subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers. It is called Adaptive Boosting as the weights are re-assigned to each instance, with higher weights assigned to incorrectly classified instances. In some problems it can be less susceptible to the overfitting problem than other learning algorithms.