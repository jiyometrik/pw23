"""
*** rq2_main.py ***
The driver code for answering RQ2 of this project.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(context="poster", font="TeX Gyre Heros")

# open the sentistrength data file
DATAPATH = os.path.join(
    os.path.dirname(__file__), "../data/"
)

df = pd.read_csv(
    os.path.join(DATAPATH, "sentistrength_data.csv")
)

SENT_POS, SENT_NEG = df["sent.pos"], df["sent.neg"]
sent_nets, polarities = [], []

for row in df.index:
    sent_net = SENT_POS[row] + SENT_NEG[row]
    sent_nets.append(sent_net)
    polarity = (
        1 if sent_net > 0 else -1 if sent_net < 0 else 0
    )
    polarities.append(polarity)
df["sent.net"] = sent_nets
df["sent.polarity"] = polarities

# write the sentiments to a new CSV file
REVIEWS = pd.read_csv(
    os.path.join(DATAPATH, "datafiniti_reviews.csv"),
    header=0,
    sep=",",
    on_bad_lines="skip",
)
combined_data = REVIEWS[
    ["reviews.rating", "reviews.title", "reviews.text"]
].copy()

column_names = [
    "sent.pos",
    "sent.neg",
    "sent.net",
    "sent.polarity",
]
for idx, name in enumerate(column_names):
    combined_data.insert(
        idx + 1, value=df[name], column=name
    )
combined_data.to_csv(
    os.path.join(DATAPATH, "combined_sentiments.csv")
)

# NOTE ** plotting **

fig, ax = plt.subplots()
LABELS_COUNTS = {
    "Positive": sum(pol == 1 for pol in polarities),
    "Negative": sum(pol == -1 for pol in polarities),
    "Neutral": sum(pol == 0 for pol in polarities),
}

SAVEPATH = os.path.join(
    os.path.dirname(__file__), "../results/poster/"
)

# pie chart: positive/negative reviews
ax.pie(
    list(LABELS_COUNTS.values())[:2],
    labels=list(LABELS_COUNTS.keys())[:2],
    autopct=lambda p: f"{p:.2f}%",
)
ax.set_title("Proportion of positive and negative reviews")
plt.savefig(os.path.join(SAVEPATH, "pie_bipartite.png"))
plt.clf()

# pie chart: positive/negative/neutral reviews
fig, ax = plt.subplots()
ax.pie(
    list(LABELS_COUNTS.values()),
    labels=list(LABELS_COUNTS.keys()),
    autopct=lambda p: f"{p:.2f}%",
)
ax.set_title(
    "Proportion of positive, negative and neutral reviews"
)
plt.savefig(os.path.join(SAVEPATH, "pie_tripartite.png"))
