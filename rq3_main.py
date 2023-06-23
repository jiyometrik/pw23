#!/usr/bin/env python
# coding: utf-8

"""
[RQ3]
can we predict the relationship between the:
- independent: frequency of tokens of a review and its polarity?
- dependent: polarity (positive / negative)
"""

# In[1]:

# data processing
import string

# plotting
import matplotlib.pyplot as plt
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


# generate a cleaned, tokenised and lemmatised version of the reviews
df["reviews.clean"] = df["reviews.text"].apply(clean_review)

# In[3]:

# extract vector representations for each review.
documents = [
    TaggedDocument(doc, [i])
    for i, doc in enumerate(df["reviews.clean"].apply(lambda x: x.split(" ")))
]

# train a doc2vec model
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each document into vec data
df_vec = df["reviews.clean"].apply(
    lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
df_vec.columns = ["vec_" + str(x) for x in df_vec.columns]
df = pd.concat([df, df_vec], axis=1)

# In[4]:

# add the term frequency - inverse document frequency values for every word
tfidf = TfidfVectorizer(min_df=10)
tfidf_result = tfidf.fit_transform(df["reviews.clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names_out())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = df.index
df = pd.concat([df, tfidf_df], axis=1)

# In[5]:

# distribution of sentiments
for polar in [-1, 1]: # positive or negative (don't consider neutrals)
    subset = df[df["sent.polarity"] == polar]
    if polar == -1:
        STATUS = "negative"
    else:
        STATUS = "positive"
    sns.distplot(subset["sent.net"], hist=False, label=STATUS)
plt.savefig("./results/rq3/distribution.png", dpi=600)
plt.clf()

# In[6]:

# is_bad: True if polarity == -1 else False
df["review.is_bad"] = df["sent.polarity"].apply(lambda x: x == -1)

# feature selection
LABEL = "review.is_bad"
ignore_cols = [
    LABEL, "sent.polarity", "sent.pos", "sent.neg", "sent.net", "index",
    "reviews.rating", "reviews.clean", "reviews.title", "reviews.text"
]
features = [col for col in df.columns if col not in ignore_cols]

# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(df[features],
                                                    df[LABEL],
                                                    test_size=0.2,
                                                    random_state=42)

# In[7]:

# train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

# show feature importance
feature_importances_df = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
# feature_importances_df.head(20)

# In[8]:

y_pred = [pred[1] for pred in rf.predict_proba(x_test)]

# false +ve rate, true +ve rate
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)

"""receiver operating characteristic:
the higher the curve above the diagonal baseline, the better the preds"""
roc_auc = auc(fpr, tpr)

# plot the roc curve -- TODO export this curve.
plt.figure(1, figsize=(15, 10))
LW = 2
plt.plot(fpr,
         tpr,
         color="darkorange",
         lw=LW,
         label=f"ROC Curve (Area: {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], lw=LW, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic of Sentiment Prediction")
plt.legend(loc="lower right")
# plt.show()
plt.savefig("./results/rq3/roc.png", dpi=600)
plt.clf()

# In[9]:

# area-under-curve precision-recall
average_precision = average_precision_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)

step_kwargs = ({
    "step": "post"
} if "step" in signature(plt.fill_between).parameters else {})

plt.figure(1, figsize=(15, 10))
plt.step(recall, precision, color="b", alpha=0.2, where="post")
plt.fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
# TODO export this curve.
plt.title(f"2-Class Precision-Recall Curve. Average Precision: {average_precision:.2f}")
plt.savefig("./results/rq3/precision_recall.png", dpi=600)
plt.clf()
