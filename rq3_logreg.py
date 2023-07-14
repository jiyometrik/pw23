#!/usr/bin/env python
# coding: utf-8

"""
[RQ3]
can we predict the relationship between the:
- independent: frequency of tokens of a review and its polarity
- dependent: polarity (positive / negative)
using a logistic regression model?
"""

# data processing
import string

# plotting
import matplotlib.pyplot as plt
import numpy as np
# natural language processing
import nltk
import pandas as pd
import seaborn as sns
from funcsigs import signature
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import train_test_split

nltk.data.path.append("/usr/share/nltk_data/")

# matplotlib things
plt.style.use("seaborn-v0_8")

# In[2]:

# import the data
df = pd.read_csv("./data/combined_sentiments.csv",
                 header=0,
                 sep=",",
                 on_bad_lines="skip")

# lemmatise


def get_wordnet_pos(tag):
    """identify each word by its part of speech
    and return that part of speech, for lemmatisation."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


# check whether there is a digit or not


def check_digits(text):
    """check whether a piece of text
    contains numerical digits."""
    return any(i.isdigit() for i in text)


# tokenise


def clean_review(review):
    """removes stop words from each review,
    then tokensises them."""
    review = str(review)
    review = review.lower() # turn into lowercase
    review = [word.strip(string.punctuation)
              for word in review.split(" ")] # remove punctuation
    # remove digits
    review = [word for word in review if not check_digits(word)]

    # remove stop words
    stop = stopwords.words("english")
    review = [token for token in review if token not in stop]
    # remove empty tokens
    review = [token for token in review if len(token) > 0]

    # tag each token with its part of speech (pos)
    pos_tags = pos_tag(review)
    review = [
        WordNetLemmatizer().lemmatize(tag[0], get_wordnet_pos(tag[1]))
        for tag in pos_tags
    ]

    # remove words with only one letter
    review = [token for token in review if len(token) > 1]
    review = " ".join(review)
    return review
# print(type(clean_review("Housekeeper kept our rooms clean. Skyline studios very spacious & modern. Lovely big bathroom with well stocked amenities. Poolside seating & Olympic-sized pool was enjoyable.")))
# print(clean_review("Housekeeper kept our rooms clean. Skyline studios very spacious & modern. Lovely big bathroom with well stocked amenities. Poolside seating & Olympic-sized pool was enjoyable."))

# generate a cleaned, tokenised and lemmatised version of the reviews
df["reviews.clean"] = df["reviews.text"].apply(clean_review)

def build_freqs(reviews):
    yslist = np.squeeze(df['sent.polarity']).tolist()
    freqs = {}
    for y, review in zip(yslist, reviews):
        for word in clean_review(review).split():
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs

# code a sigmoid function
def sigmoid(z):
	'''
	input:  z - can be a scalar or an array
	iutput: h - the sigmoid of z
	'''
	# calculate the sigmoid of z
	h = 1/(1 + np.exp(-z))
	return h

def gradient_desc(x, y, theta, alpha, iters):
	'''
	input: 
     x - matrix of features which is m * (n + 1)
		 y - corresponding labels of the input matrix x, dimensions
		 theta: weight vector of dimension (n+1,1)
		 alpha: learning rate
		 iters: number of training iterations
	'''
	m = len(x)
	for _ in range(0, iters):
		z = np.dot(x, theta)
		h = sigmoid(z)
		j = (-1/m)*(np.dot(y.T,np.log(h)) + np.dot((1-y).T,np.log(1-h)))
		theta = theta - (alpha/m)*np.dot(x.T, h-y)
	j = float(j)
	return j, theta

def extract_features(review, freqs):
	wordlist = clean_review(review).split()
	x = np.zeros((1, 3))
	x[0, 0] = 1
	for word in wordlist:
		x[0, 1] += freqs.get((word, 1), 0)
		x[0,2] += freqs.get((word,0),0)
	assert(x.shape == (1, 3))
	return x

# set training and test sets
train_pos = df[df['sent.polarity'] == 1]['reviews.clean'].tolist()[:7001]
train_neg = df[df['sent.polarity'] == -1]['reviews.clean'].tolist()[:7001]
test_pos = df[df['sent.polarity'] == 1]['reviews.clean'].tolist()[7001:]
test_neg = df[df['sent.polarity'] == -1]['reviews.clean'].tolist()[7001:]
train_x = train_pos + train_neg
test_x = test_pos + test_neg
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# finally build the logreg 
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], build_freqs(train_x)) # not defined
# training labels corresponding to X
Y = train_y
# Apply gradient descent
J, theta = gradient_desc(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")