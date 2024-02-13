import os

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper")

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string

from funcsigs import signature
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


DATADIR = os.path.join(
    os.path.abspath(os.path.dirname("")), "../data"
)
DF = pd.read_csv(
    os.path.join(DATADIR, "combined_sentiments.csv"),
    header=0,
    sep=",",
    on_bad_lines="skip",
)
STOP = stopwords.words("english")


def get_wordnet_pos(pos_tag):
    """lemmatises words by classifying them into their
    respective parts of speech."""
    if pos_tag.startswith("J"):
        return wordnet.ADJ
    elif pos_tag.startswith("V"):
        return wordnet.VERB
    elif pos_tag.startswith("N"):
        return wordnet.NOUN
    elif pos_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def check_digits(text):
    """is there a digit in the text?"""
    return any(i.isdigit() for i in text)


def clean_review(review):
    """tokenise and clean up punctuation"""
    review = str(review).lower()
    review = [
        word.strip(string.punctuation)
        for word in review.split(" ")
    ]
    review = [
        word for word in review if not check_digits(word)
    ]

    review = [
        token for token in review if token not in STOP
    ]
    review = [token for token in review if len(token) > 0]

    pos_tags = pos_tag(review)
    review = [
        WordNetLemmatizer().lemmatize(
            tag[0], get_wordnet_pos(tag[1])
        )
        for tag in pos_tags
    ]

    review = [token for token in review if len(token) > 1]
    review = " ".join(review)
    return review


DF["reviews.clean"] = DF["reviews.text"].apply(clean_review)
REVIEWS_CLEAN = DF["reviews.clean"]
REVIEWS_ALL = DF["reviews.clean"].values.tolist()


COUNT_VECTORISER = CountVectorizer()
WORDBAG = COUNT_VECTORISER.fit_transform(REVIEWS_ALL)

TFIDF_TRANSFORM = TfidfTransformer(
    use_idf=True,
    norm="l2",
    smooth_idf=True,
    sublinear_tf=True,
)
TFIDF_TRANSFORM.fit_transform(WORDBAG).toarray()

X = REVIEWS_CLEAN
y = DF["sent.polarity"]

X_t, X_test, y_t, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_t, y_t, test_size=0.25, random_state=1, stratify=y_t
)

PARAM_GRID = [
    {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [STOP, None],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [1.0, 10.0, 100.0],
    },
    {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [STOP, None],
        "vect__use_idf": [False],
        "vect__norm": [None],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [1.0, 10.0, 100.0],
    },
]


TFIDF_VECTORISER = TfidfVectorizer(
    strip_accents=None, lowercase=False, preprocessor=None
)
pipeline = Pipeline(
    [
        ("vect", TFIDF_VECTORISER),
        ("clf", LogisticRegression(random_state=42)),
    ]
)

gridsearch = GridSearchCV(
    pipeline,
    PARAM_GRID,
    scoring="accuracy",
    cv=5,
    verbose=1,
    n_jobs=-1,
)
gridsearch.fit(X_train, y_train)

print(f"best accuracy: {gridsearch.best_score_:.5f}")
clf = gridsearch.best_estimator_
print(f"accuracy in test: {clf.score(X_test, y_test):.5f}")

PREDS = clf.predict(X_val)
ACTUALS = y_val.to_numpy()
ACTUALS[ACTUALS == 0] = -1

RESULTSDIR = os.path.join(
    os.path.abspath(os.path.dirname("")), "../results/rq3"
)

FALSE_POS_RATE, TRUE_POS_RATE, thresholds = roc_curve(
    ACTUALS, PREDS, pos_label=1
)
roc_auc = auc(FALSE_POS_RATE, TRUE_POS_RATE)

plt.plot(
    FALSE_POS_RATE,
    TRUE_POS_RATE,
    label=f"ROC (Area: {roc_auc:.3f})",
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title(
    "Receiver operating characteristic (logistic regression)"
)
plt.legend(loc="best")
plt.savefig(os.path.join(RESULTSDIR, "logistic_roc.png"))

AVG_PRECISION = average_precision_score(
    ACTUALS, PREDS, pos_label=1
)
PRECISION, RECALL, _ = precision_recall_curve(
    ACTUALS, PREDS
)
step_kwargs = (
    {"step": "post"}
    if "step" in signature(plt.fill_between).parameters
    else {}
)

plt.step(
    RECALL,
    PRECISION,
    where="post",
    label=f"Precision-Recall (Avg. Precision: {AVG_PRECISION:.3f})",
)
plt.fill_between(
    RECALL, PRECISION, alpha=0.5, **step_kwargs
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.title(
    f"2-Class Precision-Recall Curve (logistic regression)"
)
plt.legend(loc="best")
plt.savefig(os.path.join(RESULTSDIR, "logistic_prc.png"))
