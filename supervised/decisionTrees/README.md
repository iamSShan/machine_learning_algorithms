## Decision Trees
 
 Decision tree is the powerful and popular tool for classification and prediction. A Decision tree is a flowchart like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.
 In order to build a tree, we use an algorithm called CART(classification and regresion tree)

### Some basic terms:

 * Root Node: Base node of the tree, it represents the entire sample and this gets classified into two or more homogeneous sets.
 * Leaf Node: We get this when we reach at the end of the tree
 * Splitting: Diving root node or sub-node to further sets based on some condition.
 * Branch/Sub Tree: It is formed, when tree or node is splitted. In other words, it is a sub section of entire tree.
 * Pruning: It is opposite of splitting. It is removing of unwanted branches from tree.
 * Impurity: For example if a bag has just blue color balls and we have to find probability that we pick blue color ball only, then in that case impurtiy is zero. But if that same bag has different color balls, then impurity will be greater than 0.
 
### Important terms:
 * Entropy: It is the measure of impurity(or purity). It defines randomness in data. It is the first step to solve the problem of decision tree. In order to select best attribute of the node, which can be used for splitting further, we use entropy.
 Entropy values ranges between 0 to 1(1 means completely impure subset, like when we have equal number of yes and no). We calculate entropy for available attributes and finally choose one lowest entropy. When we get 0 as entropy we call it as a pure sub split and then it is treated as leaf node.
 It is given by:
 ![Entropy](images/entropy.jpg)

* Information Gain: It measures the reduction in entropy, it decides which attribute should be selected as decision node. Constructing decision trees involves finding attribute that returns highest information gain. It is given by
 ![Information Gain](images/info_gain.png)

 We calculate entropy for each cases and then calculate information gain also, and then compare the information gain.
 * Gini Index: It is measure of impurity(or purity) used in build decision tree in CART.
	![Gini Impurity](images/gini_impurity.png)
 * Reduction in variance: It is an algo which is used for continuous target variable(regression problems). The split with lower variance is selected as the criteria to split the sample.
 * Chi Sqaure: It is an algo to find out the statistical significance between the differences between sub nodes and parent node.
