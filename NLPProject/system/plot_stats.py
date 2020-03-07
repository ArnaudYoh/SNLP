import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#####################################################
# Plot accuracy against sentence length
df = pd.read_csv('results/results_pyevalb_dataframe.txt', delimiter='|')
length_sentences = df[" words "].values
tag_accuracy = df[" tag_accracy "].values
hist = {}
for idx_sent in range(np.shape(length_sentences)[0]):
    length = length_sentences[idx_sent]
    if length in hist:
        hist[length].append(tag_accuracy[idx_sent])
    else:
        hist[length] = [tag_accuracy[idx_sent]]
for length in hist.keys():
    hist[length] = np.mean(hist[length])
plt.bar(list(hist.keys()), list(hist.values()))
plt.xlabel("length of sentence")
att = "postag accuracy"
plt.ylabel(att)
plt.savefig("results/fig_" + att)
plt.show()

#####################################################
# Plot number of mistakes against sentence length
df = pd.read_csv('results/results_pyevalb_dataframe.txt', delimiter='|')
length_sentences = df[" words "].values
tag_accuracy = df[" words "].values - df[" correct_tags "].values
hist = {}
for idx_sent in range(np.shape(length_sentences)[0]):
    length = length_sentences[idx_sent]
    if length in hist:
        hist[length].append(tag_accuracy[idx_sent])
    else:
        hist[length] = [tag_accuracy[idx_sent]]
for length in hist.keys():
    hist[length] = np.mean(hist[length])
plt.bar(list(hist.keys()), list(hist.values()))
plt.xlabel("length of sentence")
att = "number of postag mistakes"
plt.ylabel(att)
plt.savefig("results/fig_" + att)
plt.show()

#####################################################
# Plot recall/precision against sentence length
ax = plt.subplot(111)
w = 0.3
length_sentences = df[" words "].values
tag_accuracy = df[" prec "].values
hist = {}
for idx_sent in range(np.shape(length_sentences)[0]):
    length = length_sentences[idx_sent]
    if length in hist:
        hist[length].append(tag_accuracy[idx_sent])
    else:
        hist[length] = [tag_accuracy[idx_sent]]
for length in hist.keys():
    hist[length] = np.mean(hist[length])
ax.bar(np.array(list(hist.keys())) - w, list(hist.values()), width=w / 4., color='b', align='center')

tag_accuracy = df[" recall "].values
hist = {}
for idx_sent in range(np.shape(length_sentences)[0]):
    length = length_sentences[idx_sent]
    if length in hist:
        hist[length].append(tag_accuracy[idx_sent])
    else:
        hist[length] = [tag_accuracy[idx_sent]]
for length in hist.keys():
    hist[length] = np.mean(hist[length])
ax.bar(np.array(list(hist.keys())) + w, list(hist.values()), width=w / 4., color='r', align='center')
ax.autoscale(tight=True)

plt.xlabel("length of sentence")
att = "bracketting precision (blue), recall (red)"
plt.ylabel(att)
plt.savefig("results/fig_" + att)
