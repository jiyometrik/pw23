'''
imports.
'''
# from afinn import Afinn
from os import path
import matplotlib.pyplot as plt
import nltk as nt
import pandas as pd
import wordcloud as wc
import numpy as np

# TODO: uncomment the following two lines for the first time you run this program!
# nt.download('punkt')
# nt.download('stopwords')

# open the file.
lines = open('./data/rq2_sentistrength_data.txt', 'r').read().split('\n')
sentiments = []
for line in lines:
	terms = line.split('\t')
	positive = terms[2]
	negative = terms[3][1]
	if positive.isnumeric() and negative.isnumeric():
		sent_sum = int(positive) - int(negative)
		if sent_sum > 0:
			polarity = 1
		elif sent_sum < 0:
			polarity = -1
		elif sent_sum == 0:
			polarity = 0
		sentiments.append(tuple((int(positive), -int(negative), polarity)))
print(sentiments[:5])

sents = np.array(sentiments)
# a = np.where(sents)
# print(a)
df = pd.DataFrame(sents, columns=['pos', 'neg', 'pol'])
print(df)
positive_pairs = df.loc[df['pol'] == 1].size
negative_pairs = df.loc[df['pol'] == -1].size
neutral_pairs = df.loc[df['pol'] == 0].size

# print charts and stuff
# tripartite
fig_tri, ax_tri = plt.subplots()
labels_tri = 'Positive', 'Negative', 'Neutral'
fracs_tri = [positive_pairs, negative_pairs, neutral_pairs]
ax_tri.pie(fracs_tri, labels=labels_tri, autopct="%1.1f%%", shadow=False)
ax_tri.set_title("Proportion of Positive, Negative and Neutral Reviews")
plt.savefig("./results/rq2/pie_chart_3part.png", dpi=600)

# bipartite
fig_bi, ax_bi = plt.subplots()
labels_bi = 'Positive', 'Negative'
fracs_bi = [positive_pairs, negative_pairs]
ax_bi.pie(fracs_bi, labels=labels_bi, autopct="%1.1f%%", shadow=False)
ax_bi.set_title("Proportion of Positive and Negative Reviews")
plt.savefig("./results/rq2/pie_chart_2part.png", dpi=600)