"""
Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words,
normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, 
which is known as the lemma.
It also does same thing as stemmming, but instead it returns meaningful base words.
For e.g: history and historical will get convert to `history`
	   : final, finally and finalized will get convert to `final`
	   : going, goes and gone to go

For some of the words, stemming is not able to give meaningful word, but lemmatization gives.
Lemmatization takes usually lot of time as it does more processing compared to stemming.

Applications of Lemmatization: chatbot, question answer app // as we need meaningful response here
"""

import nltk
from nltk.stem import WordNetLemmatizer # Library for lemmatization
# Stopwords remove words which are not so important like `and`, 'of', 'for', 'or', etc.
from nltk.corpus import stopwords

paragraph = "Musk attended Queen’s University in Kingston, Ontario, and in 1992 he transferred to the University of Pennsylvania, Philadelphia, where he received bachelor’s degrees in physics and economics in 1995. He enrolled in graduate school in physics at Stanford University in California, but he left after only two days because he felt that the Internet had much more potential to change society than work in physics. That year he founded Zip2, a company that provided maps and business directories to online newspapers. In 1999 Zip2 was bought by the computer manufacturer Compaq for $307 million, and Musk then founded an online financial services company, X.com, which later became PayPal, which specialized in transferring money online. The online auction eBay bought PayPal in 2002 for $1.5 billion."

# Get the sentences
sentences = nltk.sent_tokenize(paragraph)
# Intialize and create and object of it
lemmatizer = WordNetLemmatizer()

# Lemmatization
for i in range(len(sentences)):
	# Get list of words from each sentence
	words = nltk.word_tokenize(sentences[i])
	# Remove stopwords now and stem each word
	words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words("english"))]
	# Join the obtained words and update the sentence
	sentences[i] = " ".join(words)

print(sentences)
