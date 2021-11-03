## Naive Bayes

* Dependent events: If the result of one event affects the result of other then both events are dependent. For example: 
	** Suppose we have 5 blue marbles and 5 red marbles in a bag. We pull out one marble, which may be blue or red. Now there are 9 marbles left in the bag. What is the probability that the second marble will be red?
	** It depends. If the first marble was red, then the bag is left with 4 red marbles out of 9 so the probability of drawing a red marble on the second draw is 4/9. But if the first marble we pull out of the draw is blue, then there are still 5 red marbles in the bag and the probability of pulling a red marble out of the bag is 5/9.
	** The second draw is a dependent event. It depends upon what happened in the first draw.

* Independent events: Independent events are events that do not affect the outcome of each other. In terms of probability, two events are independent if the probability of one event occurring no way affects the probability second event occurring. For example, if two coins are tossed simultaneously then they outcome of one won't affect outcome of second.

## Naive Bayes Classifier:
These are collection of classification algorithm based on Bayes theorem.

### Bayes theorem(alternatively Bayes’s law or Bayes’s rule):
	
* It is named after British mathematician Thomas Bayes, it is mathematical formula for determining conditional probablity.

* It describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if the probability that someone has cancer is related to their age, using Bayes’ theorem the age can be used to more accurately assess the probability of cancer than can be done without knowledge of the age.

* In other words, conditional probability is defined as the likelihood of an event or outcome occurring, based on the occurrence of a previous event or outcome
	- The theorem provides a way to revise existing predictions or theories (update probabilities) given new or additional evidence
	- Formula for Bayes' theorem:
	P(A|B) = P(A∩B)/P(B) = P(A).P(B|A) / P(B)
	P(A)= The probability of A occurring
	P(B)= The probability of B occurring
	P(A∣B)=The probability of A given B(how often A happens given that B happens)
	P(B∣A)= The probability of B given A(how often B happens given that A happens,)
	P(A∩B) = The probability of both A and B occurring
	
* For example, if we are pulling out balls from red and blue balls bag, we picked out red ball at first(this will be your P(B) ), now probability to pick up the next ball given previously a ball is picked up will be P(A|B).

* Bayes' theorem relies on incorporating prior probability distributions in order to generate posterior probabilities. Prior probability, in Bayesian statistical inference, is the probability of an event before new data is collected.

* Posterior probability is the revised probability of an event occurring after taking into consideration new information. Posterior probability is calculated by updating the prior probability by using Bayes' theorem. In statistical terms, the posterior probability is the probability of event A occurring given that event B has occurred.
* For eg: Let us say P(Fire) means how often there is fire, and P(Smoke) means how often we see smoke, then:

		P(Fire|Smoke) means how often there is fire when we can see smoke
	
		P(Smoke|Fire) means how often we can see smoke when there is fire

	So the formula kind of tells us "forwards" P(Fire|Smoke) when we know "backwards" P(Smoke|Fire)

	Example: If dangerous fires are rare (1%) but smoke is fairly common (10%) due to barbecues, and 90% of dangerous fires make smoke then:
	P(Fire|Smoke) =	 P(Fire).P(Smoke|Fire)/P(Smoke) 
	=	 1% x 90%/10% 
	=	9%
	So the "Probability of dangerous Fire when there is Smoke" is 9%


### Formula derivation for Bayes Theorem:
	- We know, P(A|B) = P(A∩B)/P(B)  and  P(B|A) = P(B∩A)/P(A)  (this can be proved using picking up balls from bag problem, where P(A∩B) will be given by prob of first event * prob of second event)
	- and also we know P(A∩B) = P(B∩A)
	- So we can write P(A∩B) = P(A|B)*P(B) ---- eqn(1)
	- P(B∩A) = P(B|A) * P(A)  ---- eqn(2)
	- As P(A∩B) = P(B∩A), so we can write
	- P(A|B) * P(B) = P(B|A) * P(A)
	- Therefore, P(A|B) = ( P(B|A) * P(A) ) / P(B)
	- P(A|B) is posterior probability, P(B|A) is likelihood, P(B) is marginal probability, P(A) is called prior probability


- If more than one features are given and we need to use naive bayes classifier, then we will use same formula: P(A|B) = ( P(B|A) * P(A) ) / P(B)
- Let' say features are x1, x2, x3, xn and we need to find P(y|x1,x2,x3,...xn)

- So we can write using above formula: P(y|x1,x2,x3,...xn) =  P(x1|y) P(x2|y) P(x3|y) .........P(xn|y)  * P(y)  / P(x1) P(x2) P(x3)...P(xn)

- Now on RHS, P(y) * π(i=1 to n) P(xi|y) / P(x1) P(x2) P(x3)...P(xn)  // π is product

- Now we can this P(x1) P(x2) P(x3)...P(xn) i.e. denominator as constant as it will be same for every record

- Therefore  P(y|x1,x2,x3,...xn) ∝ P(y) * π(i=1 to n) P(xi|y)   // directly proportional

- Since it is directly proportional, for finding output we need to take argmax of RHS; argmax will give the highest probability, for example of 0 has 0.5 prob and 1 has 0.3 prob, then 0.5 will be taken.
- Therefore, y = argmax P(y) * π(i=1 to n) P(xi|y)


- Naive Bayes is called `naive` because it assumes all features are mutually independent, which actually doesn't happens in real life always.
- Naive Bayes is a simple classifier known for doing well when small number of observations are available
- Naive Bayes Classifier belongs to the category of Probabilistic Classifiers. A probabilistic classifier can predict given observation by using a probability distribution over a set of classes and based on that distribution it will predict the most likely class that the observation should belong to.
- While performing Naive Bayes classification, the commonly used Probability distributions are Gaussian distribution, Multinomial Distribution, Bernoulli distribution etc.
- P(A|B) = P(A).P(B|A) / P(B) In fact, the numerator of the Bayes theorem is enough to predict the classes because the denominator is independent of any class A.
- Read more: https://medium.com/machine-learning-algorithms-from-scratch/naive-bayes-classification-from-scratch-in-python-e3a48bf5f91a


### Advantages
* Work Very well with many number of features and with Large training Dataset
* No feature scaling is required
* It converges faster when we are training the model
* It also performs well with categorical features

### Disadvantages
* Correlated features affects performance

### Outliers:
* There are different flavors of Naive Bayes, so the answer depends a bit on the use case.
* Bernoulli Naive Bayes applied to word features will always produce 0 probabilities when it encounters a word that wasn’t seen in the training data.
* However, all these and similar issues of Naive Bayes have well-known solutions (like Laplace smoothing, i.e. adding an artificial count for every word) and are routinely implemented.
* In Gaussian Naive Bayes, outliers will affect the shape of the Gaussian distribution and have the usual effects on the mean etc.
* So depending on our use case, it still makes sense to remove outliers.


### Applications:
* Sentiment Analysis
* Spam classification
* Twitter sentiment analysis
* Document categorization


* Naive Bayes are mostly used in natural language processing (NLP) problems. Naive Bayes predict the tag of a text. They calculate the probability of each tag for a given text and then output the tag with the highest one. 