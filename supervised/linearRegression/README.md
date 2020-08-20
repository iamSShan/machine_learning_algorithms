## Linear Regression(Regression algorithm)

* Linear Regression is a method used to define a relationship between a dependent variable (Y) and independent variable (X). Which is simply written as : y = mx + b ; Where y is the dependent variable, m is the scale factor or coefficient, b being the bias coefficient and X being the independent variable.(https://stats.stackexchange.com/questions/13643/what-intuitively-is-bias)

* The bias coefficient gives an extra degree of freedom to this model. The goal is to draw the line of best fit between X and Y which estimates the relationship between X and Y.

* Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables (x) and the single output variable (y). More specifically, that y can be calculated from a linear combination of the input variables (x).

* Linear combination of input means -> Suppose we have inputs: x1, x2 and x3 then it is: ax1 + bx2 + cx3

* Linear combination of vectors means ->  A linear combination of vectors is a sum of scalar multiples of those vectors. That is, given a set of M vectors xi of the same type, such as R^N (they must have the same number of elements so they can be added), a linear combination is formed by multiplying each vector by a scalar `αi` (alpha) and summing to produce a new vector y of the same type:   y = α1.x1 + α2.x2 +α3.x3 + ....αm.xm 

* Linear Regression is a supervised machine learning algorithm where the predicted output is continuous and has a constant slope. It’s used to predict values within a continuous range, (e.g. sales, price) rather than trying to classify them into categories (e.g. cat, dog). There are two main types: When there is a single input variable (x), the method is referred to as simple linear regression. When there are multiple input variables, literature from statistics often refers to the method as multiple linear regression(multivariable regression).
	
	Simple linear regression: Simple linear regression uses traditional slope-intercept form, where m and b are the variables our algorithm will try to “learn” to produce the most accurate predictions. x represents our input data and y represents our prediction.   y=mx+b

	Multivariable regression: A more complex, multi-variable linear equation might look like this, where w represents the coefficients, or weights, our model will try to learn. f(x,y,z)=w1x+w2y+w3z

	Read more here: https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
				  : https://towardsdatascience.com/coding-deep-learning-for-beginners-linear-regression-part-1-initialization-and-prediction-7a84070b01c8

- There can be multiple lines that can minimize absolute error, but there will be only one line that will minimize sum of squared errors(SSE). Therefore using SSE also makes implementation easier.
- Best regression is the one that minimizes sum of squared errors. Σ(actual-predicted)^2
- Several algo for for minimizing sum of squared errors:
	* ordinary least squares
	* gradient descent
- Event though, there is shortcoming with SSE, as data points increases, SSE increase. Which means if a line fitting less data points in a less efficient than a line fitting more data point dataset. Former will have less SSE compared to latter.
- Evaluation metric which does not has this shortcoming is R squared
- R^2 answers the question: how much of change in the output(y) is explained by the change in the input(X)
- This number should be bigger.

- Different techniques can be used to prepare or train the linear regression equation from data, the most common of which is called Ordinary Least Squares. It is common to therefore refer to a model prepared this way as Ordinary Least Squares Linear Regression or just Least Squares Regression.
- The linear equation assigns one scale factor to each input value or column, called a coefficient and represented by the capital Greek letter Beta (B). One additional coefficient is also added, giving the line an additional degree of freedom (e.g. moving up and down on a two-dimensional plot) and is often called the intercept or the bias coefficient.(y = B0 + B1*x)
- It is common to talk about the complexity of a regression model like linear regression. This refers to the number of coefficients used in the model.
- When a coefficient becomes zero, it effectively removes the influence of the input variable on the model and therefore from the prediction made from the model (0 * x = 0). This becomes relevant if you look at regularization methods that change the learning algorithm to reduce the complexity of regression models by putting pressure on the absolute size of the coefficients, driving some to zero.
- When we will want to have a brief look at four techniques to prepare a linear regression model. But actually, there are many more techniques because the model is so well studied. Take note of Ordinary Least Squares because it is the most common method used in general. Also take note of Gradient Descent as it is the most common technique taught in machine learning classes.
 - Simple linear regression:
 	* With simple linear regression when we have a single input, we can use statistics to estimate the coefficients.
 	* This requires that you calculate statistical properties from the data such as means, standard deviations, correlations and covariance. All of the data must be available to traverse and calculate statistics.
 	* This is fun as an exercise in excel, but not really useful in practice.

 - Ordinary Least Squares:
 	* When we have more than one input we can use Ordinary Least Squares to estimate the values of the coefficients.
 	* The Ordinary Least Squares procedure seeks to minimize the sum of the squared residuals. This means that given a regression line through the data we calculate the distance from each data point to the regression line, square it, and sum all of the squared errors together. This is the quantity that ordinary least squares seeks to minimize.
 - Gradient Descent:
 	* When there are one or more inputs you can use a process of optimizing the values of the coefficients by iteratively minimizing the error of the model on your training data.
 	* This operation is called Gradient Descent and works by starting with random values for each coefficient. The sum of the squared errors are calculated for each pair of input and output values. A learning rate is used as a scale factor and the coefficients are updated in the direction towards minimizing the error. The process is repeated until a minimum sum squared error is achieved or no further improvement is possible.
 - Regularization:
 	* There are extensions of the training of the linear model called regularization methods. These seek to both minimize the sum of the squared error of the model on the training data (using ordinary least squares) but also to reduce the complexity of the model (like the number or absolute size of the sum of all coefficients in the model).
 	* Two popular examples of regularization procedures for linear regression are:
    	** Lasso Regression: where Ordinary Least Squares is modified to also minimize the absolute sum of the coefficients (called L1 regularization).
    	** Ridge Regression: where Ordinary Least Squares is modified to also minimize the squared absolute sum of the coefficients (called L2 regularization).

### Cost function of linear regression:
- It is a function that measures the performance of ML model for a given data. Cost Function quantifies the error between predicted values and expected values and presents it in the form of a single real number.
The purpose of Cost Function is to be either:

    * Minimized - then returned value is usually called cost, loss or error. The goal is to find the values of model parameters for which Cost Function return as small number as possible.
    * Maximized - then the value it yields is named a reward. The goal is to find values of model parameters for which returned number is as large as possible.
Read Tailoring Cost functions: https://towardsdatascience.com/coding-deep-learning-for-beginners-linear-regression-part-2-cost-function-49545303d29f

### Gradient Descent of Linear Regression:
- Gradient descent is an algorithm that is used to minimize a cost function. Gradient descent is used not only in linear regression; it is a more general algorithm.
- While training the model, the model calculates the cost function which measures the Root Mean Squared error between the predicted value (pred) and true value (y). The model targets to minimize the cost function.
-  To minimize the cost function, the model needs to have the best value of θ1 and θ2. Initially model selects θ1 and θ2 values randomly and then itertively update these value in order to minimize the cost function untill it reaches the minimum. By the time model achieves the minimum cost function, it will have the best θ1 and θ2 values. Using these finally updated values of θ1 and θ2 in the hypothesis equation of linear equation, model predicts the value of x in the best manner it can.
- We will start off by some initial guesses for the values of θ0 and θ1 and then keep on changing the values according to the formula:  θj:=θj − α*(∂/∂θj) * f(θ0,θ1) for j=0,1
- α is called the learning rate, and it determines how big a step needs to be taken when updating the parameters. The learning rate is always a positive number.
- We want to simultaneously update θ0 and θ1, that is, calculate the right-hand-side of the above equation for both j=0 as well as j=1 and then update the values of the parameters to the newly calculated ones, which means first calculate for both and then assign(as shown in picture below). This process is repeated till convergence is achieved.
![grad_desc_calc](images/grad_desc_calc.png)

- If α is too small, then gradient descent can be slow; if it is too large gradient descent can overshoot the minimum. It may fail to converge, or even diverge.
- Suppose θ1​ is at a local optimum of J(θ1​)(at a minimum position), what will one step of gradient descent θ1:=θ1 − α*(∂/∂θ1).J(θ1​) do? (It will be unchanged, as slope at that point will be 0, hence derivative term will be 0)
- As we approach local minimum, gradient descent will automatically take smaller steps(as derivative term becomes smaller). So there is no need to decrease α over time.
- Gradient descent is guaranteed to find the global minimum for any function J(θ0,θ1)
- Read this for further info: https://www.hackerearth.com/blog/developers/gradient-descent-algorithm-linear-regression
