import string

import matplotlib.pyplot as plt
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


documents = [
    TaggedDocument(
        doc, [i]) for i, doc in enumerate(
            df["reviews.clean"].apply(
                lambda x: x.split(' ')))]

model = Doc2Vec(
    documents,
    vector_size=5,
    window=2,
    min_count=1,
    workers=4)

df_vec = df['reviews.clean'].apply(
    lambda x: model.infer_vector(
        x.split(' '))).apply(
            pd.Series)
df_vec.columns = ['vec_' + str(x) for x in df_vec.columns]
df = pd.concat([df, df_vec], axis=1)


tfidf = TfidfVectorizer(min_df=10)
tfidf_result = tfidf.fit_transform(df['reviews.clean']).toarray()
tfidf_df = pd.DataFrame(
    tfidf_result,
    columns=tfidf.get_feature_names_out())
tfidf_df.columns = ['word_' + str(x) for x in tfidf_df.columns]
tfidf_df.index = df.index
df = pd.concat([df, tfidf_df], axis=1)


plt.title('Distribution of Reviews by Sentiment Score')
plt.xlabel('Sentiment Score')
plt.ylabel('Number of Reviews')

subset_neg = df[df['sent.polarity'] == -1]
sns.histplot(subset_neg['sent.net'], label='negative', kde=True)

subset_pos = df[df['sent.polarity'] == 1]
sns.histplot(subset_pos['sent.net'], label='positive', kde=True)
plt.savefig(
    "./results/rq3/distribution.png",
    dpi=1200,
    bbox_inches='tight')
plt.clf()


df['review.is_bad'] = df['sent.polarity'].apply(lambda x: x == -1)

LABEL = 'review.is_bad'
ignore_cols = [
    LABEL,
    "sent.polarity",
    "sent.pos",
    "sent.neg",
    "sent.net",
    "index",
    "reviews.rating",
    "reviews.clean",
    "reviews.title",
    "reviews.text"]
features = [col for col in df.columns if col not in ignore_cols]

x_train, x_test, y_train, y_test = train_test_split(
    df[features], df[LABEL], test_size=0.2, random_state=42)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

feature_importances_df = pd.DataFrame(
    {
        'feature': features,
        'importance': rf.feature_importances_}).sort_values(
            'importance',
    ascending=False)


y_pred = [pred[1] for pred in rf.predict_proba(x_test)]
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize=(15, 10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Random Forest)')
plt.legend(loc="lower right")
plt.show()
plt.savefig("./results/rq3/roc_rf.png", dpi=1200, bbox_inches='tight')


average_precision = average_precision_score(y_test, y_pred)

precision, recall, _ = precision_recall_curve(y_test, y_pred)
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
    'Precision-Recall Curve (Random Forest) / Avg. Precision: {0:0.2f}'
    .format(average_precision))
plt.savefig(
    "./results/rq3/prec-recall_rf.png",
    dpi=1200,
    bbox_inches='tight')
