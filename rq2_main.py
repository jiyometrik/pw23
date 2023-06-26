# imports
import matplotlib.pyplot as plt
import nltk as nt
import numpy as np
import pandas as pd
import wordcloud as wc

# matplotlib things
plt.figure(figsize=(3, 6), dpi=60)
plt.style.use("seaborn-v0_8")

# nt.download('punkt')
# nt.download('stopwords')

# open the file.
df = pd.read_csv("./data/sentistrength_data.csv")
positives, negatives = df["sent.pos"], df["sent.neg"]
nets, polarities = [], []
for row in df.index:
    net_sentiment = positives[row] + negatives[row]
    nets.append(net_sentiment)
    polarity = 2
    if net_sentiment > 0:
        polarity = 1
    elif net_sentiment < 0:
        polarity = -1
    elif net_sentiment == 0:
        polarity = 0
    polarities.append(polarity)

# write the sentiments to a new csv
reviews = pd.read_csv("./data/datafiniti_reviews.csv",
                      header=0,
                      sep=',',
                      on_bad_lines="skip")
combined_data = reviews[["reviews.rating", "reviews.title",
                         "reviews.text"]].copy()
combined_data.insert(1, value=df["sent.pos"], column="sent.pos")
combined_data.insert(2, value=df["sent.neg"], column="sent.neg")
combined_data.insert(3, value=nets, column="sent.net")
combined_data.insert(4, value=polarities, column="sent.polarity")
combined_data.to_csv("./data/combined_sentiments.csv")

positive_no = sum(pol == 1 for pol in polarities)
neutral_no = sum(pol == 0 for pol in polarities)
negative_no = sum(pol == -1 for pol in polarities)

# print charts and stuff
# tripartite
fig_tri, ax_tri = plt.subplots()
labels_tri = "Positive", "Negative", "Neutral"
fracs_tri = [positive_no, negative_no, neutral_no]
total_tri = sum(fracs_tri)
ax_tri.pie(
    fracs_tri,
    labels=labels_tri,
    autopct=lambda pct: f"{pct:.2f}% ({(pct * total_tri / 100):,.0f})",
    shadow=False)
ax_tri.set_title("Proportion of Positive, Negative and Neutral Reviews")
plt.savefig("./results/rq2/pie_chart_3part.png", dpi=600)
plt.clf()

# bipartite
fig_bi, ax_bi = plt.subplots()
labels_bi = "Positive", "Negative"
fracs_bi = [positive_no, negative_no]
total_bi = positive_no + negative_no
ax_bi.pie(
    fracs_bi,
    labels=labels_bi,
    autopct=lambda pct: f"{pct:.2f}% ({(pct * total_bi / 100):,.0f})",
    shadow=False)
ax_bi.set_title("Proportion of Positive and Negative Reviews")
plt.savefig("./results/rq2/pie_chart_2part.png", dpi=600)
