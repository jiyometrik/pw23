from os import path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk as nt
import pandas as pd
import wordcloud as wc
from afinn import Afinn

nt.download("punkt")
nt.download("stopwords")

plt.style.use("seaborn-v0_8-pastel")

stop = nt.corpus.stopwords.words("english")
for i in "$-@_.&+#!*\\(),'\"?:%":
    stop.append(i)
stop.append("n\'t")

data = pd.read_csv("./data/datafiniti_reviews.csv",
                   header=0,
                   sep=',',
                   on_bad_lines="skip")

bodies = data["reviews.text"].astype(str)
titles = data["reviews.title"].astype(str)
bodies = bodies.str.replace(
    "((Bad|Good):)|(\\.\\.\\. More)",
    "",
    regex=True)

bodies_tokens = (bodies.apply(nt.word_tokenize)).apply(
    lambda x: [token for token in x if token.lower() not in stop])

bodies_tokens_raw = []
for bodies_sentence in bodies_tokens:
    for bodies_token in bodies_sentence:
        bodies_tokens_raw.append(bodies_token)

tokens_sentiments = []

afn = Afinn()

for token in bodies_tokens_raw:
    tokens_sentiments.append(tuple((token, afn.score(token))))
sentiments_pos, sentiments_neg, sentiments_neu = [], [], []
for token_sentiment in tokens_sentiments:
    if token_sentiment[1] > 0:
        sentiments_pos.append(token_sentiment)
    elif token_sentiment[1] < 0:
        sentiments_neg.append(token_sentiment)
    else:
        sentiments_neu.append(token_sentiment)

tokens_pos = "".join(
    token_pos[0] +
    " " for token_pos in sentiments_pos)
tokens_neg = "".join(
    token_neg[0] +
    " " for token_neg in sentiments_neg)

totals_bi = [len(sentiments_pos), len(sentiments_neg)]
totals_tri = [
    len(sentiments_pos),
    len(sentiments_neg),
    len(sentiments_neu)]
total_bi = sum(totals_bi)
total_tri = sum(totals_tri)
labels_bi = ["Positive", "Negative"]
labels_tri = ["Positive", "Negative", "Neutral"]

figure, axes = plt.subplots()
bars_container = axes.bar(labels_bi, totals_bi)
axes.set_title("Sentiments (Token-Based, Bipartite)")
axes.set_xlabel("Sentiment (Bipartite)")
axes.set_ylabel("Number of Tokens")
axes.bar_label(bars_container, fmt="{:,.0f}")
plt.savefig("./results/rq1/bar_bipartite.png", dpi=600)

figure, axes = plt.subplots()
bars_container = axes.bar(labels_tri, totals_tri)
axes.set_title("Sentiments (Token-Based, Tripartite)")
axes.set_xlabel("Sentiment (Tripartite)")
axes.set_ylabel("Number of Tokens")
axes.bar_label(bars_container, fmt="{:,.0f}")
plt.savefig("./results/rq1/bar_tripartite.png", dpi=600)

fig_pie_bi, ax_pie_bi = plt.subplots()
ax_pie_bi.set_title(
    "Proportion of Tokens by Sentiment; Positive v. Negative")
ax_pie_bi.pie(
    totals_bi,
    labels=labels_bi,
    autopct=lambda pct: "{:.2f}% ({:,.0f})".format(
        pct,
        pct *
        total_bi /
        100),
    shadow=False)

plt.savefig(
    "./results/rq1/pie_bipartite.png",
    dpi=1200,
    bbox_inches='tight')

fig_pie_tri, ax_pie_tri = plt.subplots()
ax_pie_tri.set_title("""Proportion of Tokens by Sentiment;
                     Positive v. Negative v. Neutral""")
ax_pie_tri.pie(
    totals_tri,
    labels=labels_tri,
    autopct=lambda pct: "{:.2f}% ({:,.0f})".format(
        pct,
        pct *
        total_tri /
        100),
    shadow=False)
plt.savefig(
    "./results/rq1/pie_tripartite.png",
    dpi=1200,
    bbox_inches='tight')

wordcloud = wc.WordCloud(background_color="white",
                         mode="RGB",
                         width=1280,
                         height=720)
wordcloud.generate(tokens_pos)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
wordcloud.to_file("./results/rq1/wordcloud_pos.png")

wordcloud.generate(tokens_neg)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
wordcloud.to_file("./results/rq1/wordcloud_neg.png")
