# coding: utf-8

import nltk

paragraph = "Musk attended Queen’s University in Kingston, Ontario, and in 1992 he transferred to the University of Pennsylvania, Philadelphia, where he received bachelor’s degrees in physics and economics in 1995. He enrolled in graduate school in physics at Stanford University in California, but he left after only two days because he felt that the Internet had much more potential to change society than work in physics. That year he founded Zip2, a company that provided maps and business directories to online newspapers. In 1999 Zip2 was bought by the computer manufacturer Compaq for $307 million, and Musk then founded an online financial services company, X.com, which later became PayPal, which specialized in transferring money online. The online auction eBay bought PayPal in 2002 for $1.5 billion."


# sent_tokenize applies many regular expression on the given paragraph and converts it into different different sentences
# Stores in form of list
sentences = nltk.sent_tokenize(paragraph)
print(sentences, "\n")
# To convert this paragraph to words, it also saves as list
# Punctuations will be also considered as a single word
words = nltk.word_tokenize(paragraph)
print(words)
