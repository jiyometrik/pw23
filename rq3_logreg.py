#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# natural language processing
import nltk
nltk.data.path.append('/usr/share/nltk_data/')
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# machine learning imports
from funcsigs import signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

# matplotlib things
plt.style.use('seaborn-v0_8-poster')


# In[2]:


# import data
df = pd.read_csv('./data/combined_sentiments.csv', header=0, sep=',', on_bad_lines='skip')

# lemmatise
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# check whether there is a digit or not
def check_digits(text):
     return any(i.isdigit() for i in text)

# tokenise
def clean_review(review):
	review = str(review)
	review = review.lower() # turn into lowercase
	review = [word.strip(string.punctuation) for word in review.split(' ')] # remove punctuation
	review = [word for word in review if not check_digits(word)] # remove digits

	# remove stop words
	stop = stopwords.words('english')
	review = [token for token in review if token not in stop]
	# remove empty tokens
	review = [token for token in review if len(token) > 0]
	
	# tag each token with its part of speech (pos)
	pos_tags = pos_tag(review)
	review = [WordNetLemmatizer().lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in pos_tags]

	# remove words with only one letter
	review = [token for token in review if len(token) > 1]
	review = ' '.join(review)
	return review

# generate a cleaned, tokenised and lemmatised version of the reviews
df['reviews.clean'] = df['reviews.text'].apply(lambda x: clean_review(x))

reviews = df["reviews.clean"].values.tolist()
# print(reviews[:6])

"""
* create a list with all the unique words in the whole corpus of reviews.
* construct a feature vector that contains the counts of how often each word occurs
	in the review.
"""
count_vectoriser = CountVectorizer()
wordbag = count_vectoriser.fit_transform(reviews)
print(count_vectoriser.vocabulary_)


# In[3]:


"""
raw term frequency: frequency of each token
[term frequency-inverse document frequency]
* tf - idf(t, d) = tf(t, d) idf(t, d)
> tf(t, d) is the raw term frequency;
> idf(t, d) is the inverse document frequency: log (n_d / [1 + df(d, t)])
	* n - total number of documents
  * df(t, d) - number of documents where term t appears.
"""

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)

np.set_printoptions(precision=2)

# feed the tf-idf transformer with our previously created bag
tfidf.fit_transform(wordbag).toarray()


# In[4]:


# split into train, test and validation sets
# X = df['reviews.clean']
X = reviews
y = df['sent.polarity']
X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.25, random_state=1, stratify=y_t)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# create a parameter grid for the model to pick the best params
paramgrid = [{'vect__ngram_range': [(1, 1)],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

# train the model!
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
leftright = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gridsearch = GridSearchCV(leftright, paramgrid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)
gridsearch.fit(X_train, y_train)


# In[5]:


print('best accuracy: %.3f' % gridsearch.best_score_)

clf = gridsearch.best_estimator_
print('accuracy in test: %.3f' % clf.score(X_test, y_test))


# In[6]:


# make some predictions
# print(type(X_val))
# print(X_val.shape)
# print(X_val.iloc[:,1])
# print(X_val)
preds = clf.predict(X_val)
actuals = y_val.to_numpy()
actuals[actuals == 0] = -1
print(preds[:10], actuals[:10])

false_rate, true_rate, thresholds = roc_curve(actuals, preds, pos_label=1)

'''
receiver operating characteristic:
the higher the curve above the diagonal baseline, the better the preds
'''
roc_auc = auc(false_rate, true_rate)

# plot the roc curve
plt.figure(1, figsize=(15, 10))
lw = 2
plt.plot(false_rate, true_rate, color='darkorange',
         lw=lw, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Logistic Regression)')
plt.legend(loc="lower right")
plt.show()
plt.savefig("./results/rq3/roc_logreg.png", dpi=800, bbox_inches='tight')

# area-under-curve precision-recall
average_precision = average_precision_score(actuals, preds, pos_label=1)
precision, recall, _ = precision_recall_curve(actuals, preds)
plt.clf()
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

plt.figure(1, figsize=(15, 10))
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-Class Precision-Recall Curve (Logistic Regression) - Average Precision: {0:0.2f}'.format(average_precision))
plt.savefig("./results/rq3/prec-recall_logreg.png", dpi=800, bbox_inches='tight')

