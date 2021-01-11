# -*- coding: utf-8 -*-
"""
Unigram, Bigram, Trigram:
Here is our sentence "I read a book about the history of America."
The machine wants to get the meaning of the sentence by separating it into small pieces. How should it do that?
1. It can regard words one by one. This is unigram; each word is a gram.  "I", "read", "a", "book", "about", "the", "history", "of", "America"
2. It can regard words two at a time. This is bigram (digram); each two adjacent words create a bigram.
"I read", "read a", "a book", "book about", "about the", "the history", "history of", "of America"
3. It can regard words three at a time. This is trigram; each three adjacent words create a trigram.
"I read a", "read a book", "a book about", "book about the", "about the history", "the history of", "history of America"

FastText is an extension to Word2Vec proposed by Facebook in 2016. Instead of feeding individual words into the Neural Network, 
FastText breaks words into several n-grams (sub-words). 
For instance, the tri-grams for the word apple is app, ppl, and ple (ignoring the starting and ending of boundaries of words). 
The word embedding vector for apple will be the sum of all these n-grams. After training the Neural Network, 
we will have word embeddings for all the n-grams given the training dataset.
Rare words can now be properly represented since it is highly likely that some of their n-grams also appears in other words.
"""

import nltk
import re
from gensim.models import FastText

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

"""
sentences: the list of split sentences.
size: the dimensionality of the embedding vector
window: the number of context words you are looking at
min_count: tells the model to ignore words with total count less than this number.
workers: the number of threads being used
sg: whether to use skip-gram or CBOW
"""
model_ted = FastText(sentences, size=100, window=5, min_count=1, workers=4, sg=1)

similar = model_ted.wv.most_similar("business")
"""
Even though the word "business" does not exist in the training dataset, it is still capable of figuring out related words to it.
If we try this in the Word2Vec defined previously, it would pop out error because such word does not exist in the training dataset.
Although it takes longer time to train a FastText model (number of n-grams > number of words),
it performs better than Word2Vec and allows rare words to be represented appropriately.
"""
print(similar)
