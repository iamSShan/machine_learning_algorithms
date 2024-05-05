# -*- coding: utf-8 -*-


"""
Stemming is the process of reducing infected words to their original word stem
For e.g: history and historical will be reduced to `histori`
	   : finally, final and finalized will be get reduced to `fina`
	   : going, goes and gone to go


Most common types of stemming: Porter Stemmer, Snowball Stemmer, Lancaster Stemmer
Applications of Stemming: sentiment classfier, gmail message spam classifier
Drawback: May produce word which does not have any meaning
"""
import nltk
from nltk.stem import PorterStemmer # Library for stemming
# Using stopwords to remove words which are not so important like `and`, 'of', 'for', 'or', etc.
from nltk.corpus import stopwords

paragraph = "Musk attended Queen’s University in Kingston, Ontario, and in 1992 he transferred to the University of Pennsylvania, Philadelphia, where he received bachelor’s degrees in physics and economics in 1995. He enrolled in graduate school in physics at Stanford University in California, but he left after only two days because he felt that the Internet had much more potential to change society than work in physics. That year he founded Zip2, a company that provided maps and business directories to online newspapers. In 1999 Zip2 was bought by the computer manufacturer Compaq for $307 million, and Musk then founded an online financial services company, X.com, which later became PayPal, which specialized in transferring money online. The online auction eBay bought PayPal in 2002 for $1.5 billion."

# Get the sentences
sentences = nltk.sent_tokenize(paragraph)
# Intialize and create and object of it
stemmer = PorterStemmer()

# Stemming
for i in range(len(sentences)):
	# Get list of words from each sentence
	words = nltk.word_tokenize(sentences[i])
	# Remove stopwords now and stem each word
	words = [stemmer.stem(word) for word in words if word not in set(stopwords.words("english"))]
	# Join the obtained words and update the sentence
	sentences[i] = " ".join(words)

print(sentences)

# We can also use Snowball stemmer. Read: https://www.geeksforgeeks.org/snowball-stemmer-nlp/
