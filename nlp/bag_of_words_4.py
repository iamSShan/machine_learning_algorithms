# -*- coding: utf-8 -*-

"""
Bag of words is just document matrix
After bringing sentences to same case(let's say lowercase) and removing stopwords, now for bag of words, 
we calculate frequency of each word.
Then we sort them in descending order, then we create a table having sentence as rows and words as columns.
Now for binary bag of words we just mark 1 or 0 if a particular word is present or not respectively.
Now when not considering binary state, we can write frequency instead of 1.
Here in this table we also have an output column 
where this words will be independent feature and output will be dependent feature. Then we can train the model using them.

Disadvantage of Bag of words: if let's say more than one word has 1 or in other case, more than one word has same frequency,
then we can not determine which word is more important.
We can use TF-IDF(term frequency - inverse document frequency), which is better than bag of words.
"""

import nltk
import re
# Stopwords remove words which are not so important like `and`, 'of', 'for', 'or', etc.
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # Library for stemming
from nltk.stem import WordNetLemmatizer # Library for lemmatization

paragraph = "Musk attended Queen’s University in Kingston, Ontario, and in 1992 he transferred to the University of Pennsylvania, Philadelphia, where he received bachelor’s degrees in physics and economics in 1995. He enrolled in graduate school in physics at Stanford University in California, but he left after only two days because he felt that the Internet had much more potential to change society than work in physics. That year he founded Zip2, a company that provided maps and business directories to online newspapers. In 1999 Zip2 was bought by the computer manufacturer Compaq for $307 million, and Musk then founded an online financial services company, X.com, which later became PayPal, which specialized in transferring money online. The online auction eBay bought PayPal in 2002 for $1.5 billion."


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# First step is cleaning the text

# To get list of sentences
sentences = nltk.sent_tokenize(paragraph)
print(sentences)
print("\n\n")

corpus = [] # Cleaned sentences will be stored here
for i in range(len(sentences)):
	# Remove all punctuations, and symbols and only keep letters
	keep = re.sub('[^a-zA-Z]', ' ', sentences[i]) # replaces everything except alphabets with space
	keep = keep.lower()
	keep = keep.split() # Convert into list of words
	keep = [lemmatizer.lemmatize(word) for word in keep if word not in set(stopwords.words('english'))]
	keep = ' '.join(keep)
	corpus.append(keep)

print(corpus)
print("\n\n")

# To create bag of words
from sklearn.feature_extraction.text import CountVectorizer
# Create object 
cv = CountVectorizer()
# fit_transform creates matrix
X = cv.fit_transform(corpus).toarray()
print(X)
