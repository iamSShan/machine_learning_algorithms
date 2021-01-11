# -*- coding: utf-8 -*-
"""
TF-IDF:
TF: Term Frequency
IDF: Inverse document frequency

TF = no.of repetition of words in sentence/no. of words in sentence
IDF = log(no. of sentences/no. of sentences containing the word)

Then at last we do TF * IDF

For e.g:
Sentence 1: good boy
Sentence 2: good girl
Sentence 3: boy girl good

Words	Frequency
good	3
boy	    2
girl	2


Now we will convert these sentences into vectors after applying TF-IDF method
TF:
		Sent1  Sent2  Sent3
good    1/2	   1/2    1/3
boy		1/2	   0      1/3
girl    0      1/2    1/3


IDF:
Words   IDF
good    log(3/3)=0
boy	    log(3/2)
girl    log(3/2)


Now multiply last two tables
		good    		boy    			girl
Sent 1  (1/2)*0    	 1/2* log(3/2)	    0
Sent 2   0			 0				    1/2*log(3/2)
Sent 3   0		     1/3*log(3/2)       1/3*log(3/2)

Hence we can see that, for Sent 1, `boy` value is higher compared to other words. so there is some semantic meaning
And similarly, For Sent 2 `girl` is given importance and for Sent 3: `boy` and `girl`
"""

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # Library for lemmatization

paragraph = "Musk attended Queen’s University in Kingston, Ontario, and in 1992 he transferred to the University of Pennsylvania, Philadelphia, where he received bachelor’s degrees in physics and economics in 1995. He enrolled in graduate school in physics at Stanford University in California, but he left after only two days because he felt that the Internet had much more potential to change society than work in physics. That year he founded Zip2, a company that provided maps and business directories to online newspapers. In 1999 Zip2 was bought by the computer manufacturer Compaq for $307 million, and Musk then founded an online financial services company, X.com, which later became PayPal, which specialized in transferring money online. The online auction eBay bought PayPal in 2002 for $1.5 billion."

lemmatizer = WordNetLemmatizer()


# First step is cleaning
# To get list of sentences
sentences = nltk.sent_tokenize(paragraph)
corpus = [] # Cleaned sentences will be stored here

for i in range(len(sentences)):
	# Remove all punctuations, and symbols and only keep letters
	keep = re.sub('[^a-zA-Z]', ' ', sentences[i]) # replaces everything except alphabets with space
	keep = keep.lower()
	keep = keep.split() # Convert into list of words
	keep = [lemmatizer.lemmatize(word) for word in keep if word not in set(stopwords.words('english'))]
	keep = ' '.join(keep)
	corpus.append(keep)

# Create TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
# Create object of Tfidf
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray() # It gives vector giving more values to more important words in sentences 
print(X)
