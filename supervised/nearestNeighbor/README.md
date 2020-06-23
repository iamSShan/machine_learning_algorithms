## KNN(k-nearest neighbors algorithm)


* The k-nearest neighbors (KNN) algorithm is a simple algorithm that stores all the available cases and classifies the new data or case based on similarity measure.
* 'K' denotes number of nearest neighbors which are helps to predict class of new data or testing data.
* The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.
* It is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems.
* Euclidean distance between two neighbors is calculated. Refer to this: https://www.mathsisfun.com/algebra/distance-2-points.html
* KNN are used in recommendation systems, like in Amazon it shows similar items when you have added a item on a cart.
* KNN algorithm is also a lazy learner because there is no learning phase of the model here, all of the work happens when prediction is requested.


### KNN Algorithm:
	* Load the data
	* Initialize K to your chosen number of neighbors
	* For each example in the data
		1. Calculate the distance between the query example and the current example from the data.
		2. Add the distance and the index of the example to an ordered collection
	* Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
	* Pick the first K entries from the sorted collection
	* Get the labels of the selected K entries
	* If regression, return the mean of the K labels
	* If classification, return the mode of the K labels

- The algorithm gets significantly slower as the number of examples and/or predictors/independent variables increase.
