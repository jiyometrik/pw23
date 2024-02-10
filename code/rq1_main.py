"""
*** rq1_main.py ***
The driver code for answering RQ1 of this project.
"""

import os
import string

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import wordcloud
from afinn import Afinn

plt.style.use(
    [
        "seaborn-v0_8-whitegrid",
        "seaborn-v0_8-paper",
        "seaborn-v0_8-colorblind",
    ]
)

# make a collection of stop words to exclude during tokenisation.
STOP = (
    nltk.corpus.stopwords.words("english")
    + [
        "...more",
        "bad",
        "good",
    ]  # exclude remnants from webscraping
    + list(string.punctuation)
)

# receive the data
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

# save all this new data into another CSV file for future reference
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
    os.path.dirname(__file__), "../results/rq1/"
)
LABELS = list(TOKENS_BY_POLARITY.keys())
HEIGHTS = [
    len(TOKENS_BY_POLARITY[label]) for label in LABELS
]

# plot graphs for positive/negative sentiment
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
plt.savefig(
    os.path.join(
        SAVEPATH,
        "bar_bipartite.png",
    )
)
plt.clf()

# plot bar graph for positive/negative/neutral sentiment
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
plt.savefig(
    os.path.join(
        SAVEPATH,
        "bar_tripartite.png",
    )
)
plt.clf()

# pie chart for positive/negative
FIG, AX = plt.subplots()
AX.set_title(
    "Proportion of tokens by sentiment (bipartite)"
)
AX.pie(
    HEIGHTS[:2],
    labels=None,
    autopct=lambda pct: f"{pct:.2f}\% [{pct * sum(HEIGHTS[:2]) / 100:,.0f}]",
)
AX.legend(loc="best", labels=LABELS[:2])
plt.savefig(
    os.path.join(
        SAVEPATH,
        "pie_bipartite.png",
    )
)
plt.clf()

# pie chart for positive/negative/neutral
FIG, AX = plt.subplots()
AX.set_title(
    "Proportion of tokens by sentiment (tripartite)"
)
wedges, texts, autotexts = AX.pie(
    HEIGHTS,
    labels=None,
    autopct=lambda pct: f"{pct:.2f}\% [{pct * sum(HEIGHTS) / 100:,.0f}]",
)
AX.legend(loc="best", labels=LABELS)
plt.savefig(
    os.path.join(
        SAVEPATH,
        "pie_tripartite.png",
    )
)
plt.clf()


# wordcloud (positive tokens)
WORDCLOUD = wordcloud.WordCloud(
    font_path="/usr/share/texmf/fonts/opentype/public/lm/lmsans12-regular.otf",
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

# wordcloud (negative tokens)
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
