# -*- coding -*-
"""
Some disadvantages of bag of words and TF-IDF:
	- Semantic info is not stored, as order of words may not be same.
	- TF-IDF gives importance to uncommon words.
	- In both, there is a chance of overfitting.

To overcome these problems, we use word2vec model
Here:
	- Each word is basically represented as 32 or more dimension vector instead of single number
	- Here semantic info and relation b/w different words is also preserved.

There are 2 type of Word2Vec:
	- Continous bag of words
	- Skipgram

Steps to create Word2vec:
 - Tokenization of sentences
 - Create histograms
 - Take most frequent words
 - Create a matrix of all unique words. It also represent the occurence relation b/w the words.
"""

import nltk
import re
from gensim.models import Word2Vec  # gensim library contains Word2Vec
from nltk.corpus import stopwords

paragraph = "Musk had long been interested in the possibilities of electric cars, and in 2004 he became one of the major funders of Tesla Motors (later renamed Tesla), an electric car company founded by entrepreneurs Martin Eberhard and Marc Tarpenning. In 2006 Tesla introduced its first car, the Roadster, which could travel 245 miles (394 km) on a single charge. Unlike most previous electric vehicles, which Musk thought were stodgy and uninteresting, it was a sports car that could go from 0 to 60 miles (97 km) per hour in less than four seconds. In 2010 the company’s initial public offering raised about $226 million. Two years later Tesla introduced the Model S sedan, which was acclaimed by automotive critics for its performance and design. The company won further praise for its Model X luxury SUV, which went on the market in 2015. The Model 3, a less-expensive vehicle, went into production in 2017."

# Pre-processing the data
text = re.sub(r'\[0-9]"\]', ' ', paragraph)
text = re.sub(r'\s+',' ', text)  # Removing unecessary spaces 
text = text.lower()  # Converting to lower case
text = re.sub(r'\d',' ', text)
text = re.sub(r'\s+',' ', text)  # Removing extra spaces 

# Get list of sentences
sentences = nltk.sent_tokenize(text)
# print(sentences)
# Convert sentences into list of list of words
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
# print(sentences)

# Remove stopwords now
for i in range(len(sentences)):
	sentences[i] = [word for word in sentences[i] if word not in stopwords.words("english")]


# Now train the Word2Vec model
model = Word2Vec(sentences, min_count=1)  # min_count indicates if word is present less than 1, then skip the word
# Usually min_count value is kept as 2, but here data is small so we want to keep it as 1.

# Now find out vocalbulary that have been find out in this model
words = model.wv.vocab  # It gives dict, with keys as words and value as gensim vector object(some dimensions, vectors)
# print(words)

# Finding vector and relationship of any word
# Like for word 'travel'
vector = model.wv["travel"]  # It will give vector with 100 dimensions
# print(vector)

# Now if we have to find out similar words like 'travel' in the paragraph
similar = model.wv.most_similar("travel")  # It gives list of tuples of words
print(similar)
