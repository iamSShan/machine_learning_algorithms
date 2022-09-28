## Clustering

* Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters.

* For e.g: Suppose, you are the head of a rental store and wish to understand preferences of your customers to scale up your business. Is it possible for you to look at details of each customer and devise a unique business strategy for each one of them? Definitely not. But, what you can do is to cluster all of your customers into say 10 groups based on their purchasing habits and use a separate strategy for customers in each of these 10 groups. And this is what we call clustering.

* Broadly, clustering can be divided into two subgroups :

    * Hard Clustering: In hard clustering, each data point either belongs to a cluster completely or not. For example, in the above example each customer is put into one group out of the 10 groups.
    * Soft Clustering: In soft clustering, instead of putting each data point into a separate cluster, a probability or likelihood of that data point to be in those clusters is assigned. For example, from the above scenario each customer is assigned a probability to be in either of 10 clusters of the retail store.

* There are many clustering algortihms, but some of the famous ones are:
	* K Means Clustering
	* Hierarchical Clustering
	* Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

* K-means is sensitive to outliers.
* Also read(for Kmeans ++): https://www.geeksforgeeks.org/ml-k-means-algorithm/
* https://www.youtube.com/watch?v=XtE7hqFsYc4

#### K-medians
* We can use K-medians, as medians are less senstive to outliers

#### K-medoid
* It is a clustering algo and it is an improved version of K-means.
* Medoid means least dissimilar object which means most similar object.
* https://www.youtube.com/watch?v=AUriFHKw0TU