## Gradient Boosting:

* Let's take an example, let's say we have two independent features f1, f2 and a dependent feature 'salary' in a variable y
* Now our first step here will be to compute base model which will give one output. Now output will give average of all salary column rows. Let' say average comes any value 'X'. We will populate 'X' in a new column called ŷ for every row
* Then in second step we will be computing residuals(or we can it as errors or pseudo residuals)
* Then we calculate loss, there are several loss functions, we can use a simple formula for understanding: (ŷ-y)^2
* Then salary will be subtracted from ŷ and values will be stored in R1 column
* Then in next step we will construct a decision tree, giving input as: our independent features i.e f1 and f2, but our target feature will be R1 instead of salary. Then this base learning model will be trained on this data.
* Now when decision tree is trained for R1 feature we will get prediction values for R1 only, let' say we will store them in a new column and it as R2.

* Now we have two outputs-one from base model that gives average and second from decision tree, if we let' assume for a record:
		y = 50 (in 1000s)
		ŷ = 75
		R1 = -25
		R2 = -23
* So if we add both model scores: 75+(-23) we get 52 and if compare it will actual label i.e. 50. This value is very very near, so does it mean our model is performing very well. Answer is No, as this is is overfitting problem, even if we get 50, this is also overfitting, as here we have low bias but high variance, So for a new dataset, it will give high value.
* So, to prevent this we will use learning rate α, when we are adding both model results
		=> 75 + α (-23) ; α ranges between 0 to 1 and is decided using hyperparameter tuning
		=> Here is α = 0.1 => 75 + 0.1 * (-23) = 72.7
* Now we see that 72.7 has huge difference compared to output salary, so we will add one more decision tree. This decision tree will be created based on the R2 value.
* Generalized formula: F(x) = h0(x) + α1.h1(x) + α2.h2(x) + .... αn.hn(x) => Σ(i=1 to n)αi.hi(x)
		Here h0x will be is base model which gave 75 as output , α1.h1(x) is first decision tree output, α2.h2(x) is second decision tree output and till n; n tell how many decision tree models are there

* It is called boosting because we are adding decision tree model sequentially after the base model, we are boosting the model based on residual model until error is reduced.