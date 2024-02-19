"""
*** rq1_main.py ***
The driver code for answering RQ1 of this project.
"""

import os
import string

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import wordcloud
from afinn import Afinn

sns.set_theme(context="poster", font="TeX Gyre Heros")

# words to be excluded
STOP = (
    nltk.corpus.stopwords.words("english")
    + [
        "...more",
        "bad",
        "good",
    ]  # exclude remnants from webscraping
    + list(string.punctuation)
)

# load the data
DATA = pd.read_csv(
    os.path.join(
        os.path.dirname(__file__),
        "../data/datafiniti_reviews.csv",
    ),
    header=0,
    sep=",",
    on_bad_lines="skip",
)

# extract the titles and bodies of all of the reviews
TITLES, BODIES = DATA["reviews.title"].astype(str), DATA[
    "reviews.text"
].astype(str).str.replace(
    "((Bad|Good):)|(\\.\\.\\. More)", "", regex=True
)

# tokenise each review and remove stop words
DATA["reviews.tokens"] = BODIES.apply(
    nltk.wordpunct_tokenize
).apply(
    lambda review: [
        token.lower()
        for token in review
        if token.lower() not in STOP
    ]
)

# start an Afinn instance to begin sentiment scoring
AFINN = Afinn()

# score each token and save scores in a new column
DATA["reviews.scores"] = DATA["reviews.tokens"].apply(
    lambda review: [
        (token, AFINN.score(token)) for token in review
    ]
)

# save all data into another CSV file for future use
DATA.to_csv(
    os.path.join(
        os.path.dirname(__file__),
        "../data/afinn_scores.csv",
    )
)

# loop through all the tokens and create lists
TOKENS_BY_POLARITY = {
    "Positive": [],
    "Negative": [],
    "Neutral": [],
}
for _, pairs in DATA["reviews.scores"].items():
    TOKENS_BY_POLARITY["Positive"] += list(
        filter(lambda pair: pair[1] > 0.0, pairs)
    )
    TOKENS_BY_POLARITY["Negative"] += list(
        filter(lambda pair: pair[1] < 0.0, pairs)
    )
    TOKENS_BY_POLARITY["Neutral"] += list(
        filter(lambda pair: pair[1] == 0.0, pairs)
    )

FIG, AX = plt.subplots()
SAVEPATH = os.path.join(
    os.path.dirname(__file__), "../results/poster/"
)
LABELS = list(TOKENS_BY_POLARITY.keys())
HEIGHTS = [
    len(TOKENS_BY_POLARITY[label]) for label in LABELS
]

# bar graph: positive/negative sentiment
container = AX.bar(LABELS[:2], HEIGHTS[:2], align="center")
AX.set_title("Token polarities (bipartite)")
AX.set_xlabel("Polarity")
AX.set_ylabel("Number of tokens")
AX.bar_label(container, fmt="{:,.0f}")
plt.ticklabel_format(
    style="sci",
    axis="y",
    scilimits=(0, 0),
    useMathText=True,
    useOffset=False,
)
plt.tight_layout()
plt.savefig(
    os.path.join(
        SAVEPATH,
        "bar_bipartite.png",
    )
)
plt.clf()

# bar graph: positive/negative/neutral sentiment
FIG, AX = plt.subplots()
container = AX.bar(LABELS, HEIGHTS, align="center")
AX.set_title("Token polarities (tripartite)")
AX.set_xlabel("Polarity")
AX.set_ylabel("Number of tokens")
AX.bar_label(container, fmt="{:,.0f}")
plt.ticklabel_format(
    style="sci",
    axis="y",
    scilimits=(0, 0),
    useMathText=True,
    useOffset=False,
)
plt.tight_layout()
plt.savefig(
    os.path.join(
        SAVEPATH,
        "bar_tripartite.png",
    )
)
plt.clf()

# wordcloud (positive)
WORDCLOUD = wordcloud.WordCloud(
    font_path="arial",
    background_color="white",
    colormap="viridis",
    width=1280,
    height=720,
)
POSITIVE_TOKENS = " ".join(
    pair[0] for pair in TOKENS_BY_POLARITY["Positive"]
)
WORDCLOUD.generate(POSITIVE_TOKENS)
plt.figure()
plt.imshow(WORDCLOUD, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud (positive tokens)")
WORDCLOUD.to_file(
    os.path.join(
        SAVEPATH,
        "wordcloud_1.png",
    )
)
plt.clf()

# wordcloud (negative)
NEGATIVE_TOKENS = " ".join(
    pair[0] for pair in TOKENS_BY_POLARITY["Negative"]
)
WORDCLOUD.generate(NEGATIVE_TOKENS)
plt.figure()
plt.imshow(WORDCLOUD, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud (negative tokens)")
WORDCLOUD.to_file(
    os.path.join(
        SAVEPATH,
        "wordcloud_0.png",
    )
)
plt.clf()
