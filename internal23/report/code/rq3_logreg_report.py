import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from funcsigs import signature
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

nltk.data.path.append('/usr/share/nltk_data/')

plt.style.use('seaborn-v0_8-poster')


df = pd.read_csv(
    './data/combined_sentiments.csv',
    header=0,
    sep=',',
    on_bad_lines='skip')


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


def check_digits(text):
    return any(i.isdigit() for i in text)


def clean_review(review):
    review = str(review)
    review = review.lower()
    review = [word.strip(string.punctuation)
              for word in review.split(' ')]
    review = [word for word in review if not check_digits(word)]

    stop = stopwords.words('english')
    review = [token for token in review if token not in stop]
    review = [token for token in review if len(token) > 0]

    pos_tags = pos_tag(review)
    review = [
        WordNetLemmatizer().lemmatize(
            tag[0], get_wordnet_pos(
                tag[1])) for tag in pos_tags]

    review = [token for token in review if len(token) > 1]
    review = ' '.join(review)
    return review


df['reviews.clean'] = df['reviews.text'].apply(
    lambda x: clean_review(x))

reviews = df["reviews.clean"].values.tolist()
count_vectoriser = CountVectorizer()
wordbag = count_vectoriser.fit_transform(reviews)
print(count_vectoriser.vocabulary_)


tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)

np.set_printoptions(precision=2)

tfidf.fit_transform(wordbag).toarray()


X = reviews
y = df['sent.polarity']
X_t, X_test, y_t, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_t, y_t, test_size=0.25, random_state=1, stratify=y_t)


paramgrid = [{'vect__ngram_range': [(1, 1)],
              'clf__penalty': ['l1', 'l2'],
              'clf__C': [1.0, 10.0, 100.0]},
             {'vect__ngram_range': [(1, 1)],
              'vect__use_idf':[False],
              'vect__norm':[None],
              'clf__penalty': ['l1', 'l2'],
              'clf__C': [1.0, 10.0, 100.0]},
             ]

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


print('best accuracy: %.3f' % gridsearch.best_score_)

clf = gridsearch.best_estimator_
print('accuracy in test: %.3f' % clf.score(X_test, y_test))


preds = clf.predict(X_val)
actuals = y_val.to_numpy()
actuals[actuals == 0] = -1
print(preds[:10], actuals[:10])

false_rate, true_rate, thresholds = roc_curve(
    actuals, preds, pos_label=1)

roc_auc = auc(false_rate, true_rate)

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
plt.savefig(
    "./results/rq3/roc_logreg.png",
    dpi=800,
    bbox_inches='tight')

average_precision = average_precision_score(
    actuals, preds, pos_label=1)
precision, recall, _ = precision_recall_curve(actuals, preds)
plt.clf()
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

plt.figure(1, figsize=(15, 10))
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(
    recall,
    precision,
    alpha=0.2,
    color='b',
    **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Precision-Recall (Logistic Regression) / Avg. Precision: {0:0.2f}'
    .format(average_precision))
plt.savefig(
    "./results/rq3/prec-recall_logreg.png",
    dpi=800,
    bbox_inches='tight')
