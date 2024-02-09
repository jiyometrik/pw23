"""imports."""
import os
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import wordcloud
from afinn import Afinn
from matplotlib import colormaps

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.pad_inches"] = 1
plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = """
\\usepackage{txfonts}
"""
sns.set_style("whitegrid")
sns.set_context("paper")


# make a collection of stop words to exclude during tokenisation.
# we add "bad" and "good" because they are often left in during webscraping.
STOPWORDS = (
    nltk.corpus.stopwords.words("english")
    + ["...more", "bad", "good"]
    + list(string.punctuation)
)

# receive the data
DATA = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../data/datafiniti_reviews.csv"),
    header=0,
    sep=",",
    on_bad_lines="skip",
)

# extract the titles and bodies of all of the reviews
TITLES, BODIES = DATA["reviews.title"].astype(str), DATA["reviews.text"].astype(
    str
).str.replace("((Bad|Good):)|(\\.\\.\\. More)", "", regex=True)

# tokenise each review and remove stop words
DATA["reviews.tokens"] = BODIES.apply(nltk.wordpunct_tokenize).apply(
    lambda review: [token.lower() for token in review if token.lower() not in STOPWORDS]
)

# start an Afinn instance to begin sentiment scoring
AFINN = Afinn()

# score each token and save scores in a new column
DATA["reviews.scores"] = DATA["reviews.tokens"].apply(
    lambda review: [(token, AFINN.score(token)) for token in review]
)

# save all this new data into another CSV file for future reference
DATA.to_csv(os.path.join(os.path.dirname(__file__), "../data/afinn_scores.csv"))

# loop through all the tokens and create lists
TOKENS_BY_POLARITY = {
    "Positive": [],
    "Negative": [],
    "Neutral": [],
}
for _, pairs in DATA["reviews.scores"].items():
    TOKENS_BY_POLARITY["Positive"] += list(filter(lambda pair: pair[1] > 0.0, pairs))
    TOKENS_BY_POLARITY["Negative"] += list(filter(lambda pair: pair[1] < 0.0, pairs))
    TOKENS_BY_POLARITY["Neutral"] += list(filter(lambda pair: pair[1] == 0.0, pairs))

FIG, AX = plt.subplots()
LABELS = list(TOKENS_BY_POLARITY.keys())
HEIGHTS = [len(TOKENS_BY_POLARITY[l]) for l in LABELS]

# plot graphs for positive/negative sentiment
container = AX.bar(LABELS[:2], HEIGHTS[:2], align="center")
AX.set_title("Token polarities (bipartite)")
AX.set_xlabel("Polarity")
AX.set_ylabel("Number of tokens")
AX.bar_label(container, fmt="{:,.0f}")
plt.ticklabel_format(
    style="sci", axis="y", scilimits=(0, 0), useMathText=True, useOffset=False
)
plt.savefig(os.path.join(os.path.dirname(__file__), "../results/rq1/bar_bipartite.png"))
plt.clf()

# plot bar graph for positive/negative/neutral sentiment
FIG, AX = plt.subplots()
container = AX.bar(LABELS, HEIGHTS, align="center")
AX.set_title("Token polarities (tripartite)")
AX.set_xlabel("Polarity")
AX.set_ylabel("Number of tokens")
AX.bar_label(container, fmt="{:,.0f}")
plt.ticklabel_format(
    style="sci", axis="y", scilimits=(0, 0), useMathText=True, useOffset=False
)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "../results/rq1/bar_tripartite.png")
)
plt.clf()

# pie chart for positive/negative
FIG, AX = plt.subplots()
AX.set_title("Proportion of tokens by sentiment (bipartite)")
AX.pie(
    HEIGHTS[:2],
    labels=None,
    autopct=lambda percent: f"{percent:.2f}\% [{percent * sum(HEIGHTS[:2]) / 100:,.0f}]",
)
AX.legend(loc="best", labels=LABELS[:2])
plt.savefig(os.path.join(os.path.dirname(__file__), "../results/rq1/pie_bipartite.png"))
plt.clf()

# pie chart for positive/negative/neutral
FIG, AX = plt.subplots()
AX.set_title("Proportion of tokens by sentiment (tripartite)")
wedges, texts, autotexts = AX.pie(
    HEIGHTS,
    labels=None,
    autopct=lambda percent: f"{percent:.2f}\% [{percent * sum(HEIGHTS) / 100:,.0f}]",
)
AX.legend(loc="best", labels=LABELS)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "../results/rq1/pie_tripartite.png")
)
plt.clf()


# wordcloud (positive tokens)
WORDCLOUD = wordcloud.WordCloud(
    font_path="/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf",
    background_color="white",
    colormap="tab10",
    width=1280,
    height=720,
)
POSITIVE_TOKENS = "".join(
    np.unique([pair[0] + " " for pair in TOKENS_BY_POLARITY["Positive"]])
)
WORDCLOUD.generate(POSITIVE_TOKENS)
plt.figure()
plt.imshow(WORDCLOUD, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud (positive tokens)")
WORDCLOUD.to_file("./results/rq1/wordcloud_pos.png")
plt.clf()

# wordcloud (negative tokens)
NEGATIVE_TOKENS = "".join(
    np.unique([pair[0] + " " for pair in TOKENS_BY_POLARITY["Negative"]])
)
WORDCLOUD.generate(NEGATIVE_TOKENS)
plt.figure()
plt.imshow(WORDCLOUD, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud (negative tokens)")
WORDCLOUD.to_file("./results/rq1/wordcloud_neg.png")
plt.clf()
