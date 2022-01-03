## XgBoost

* It stands for eXtreme gradient Boosting.
* XGBoost is a decision-tree-based ensemble machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks.
* Generally, XGBoost is fast. Really fast when compared to other implementations of gradient boosting.
* XGBoost dominates structured or tabular datasets on classification and regression predictive modeling problems.

* The evidence is that it is the go-to algorithm for competition winners on the Kaggle competitive data science platform.
* The XGBoost library implements the gradient boosting decision tree algorithm.

* Both xgboost and gbm(gradient boosting machines) follows the principle of gradient boosting. There are however, the difference in modeling details. Specifically, xgboost used a more regularized model formalization to control over-fitting, which gives it better performance.

* Adaboost was the original implementation of boosting, with a single cost function and a difficulty in adapting to different link functions to create a linear model with a given outcome. Gradient boosting generalizes the framework and allows for easier computation. It can use multiple baselearner types (trees, linear terms, splines...), and cost functions and link functions are modifiable. XGBoost uses a few computational tricks that exploit a computer's hardware to speed up gradient descent and line search components, as well as a penalty function (similar to elastic net penalties) to allow for robust, sparse modeling (which also speeds up the algorithm).