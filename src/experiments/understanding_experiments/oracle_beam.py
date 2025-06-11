    
from itertools import count
import json
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def calculate_wer(baseline, proposed):
    if(baseline ==0):
        return 0
    else:
        return (baseline-proposed)/baseline 

with open(f'../result/wer_npsc_experiment_27.json', 'r') as file:
    wer_data_points = json.load(file)

best_wer = []
actual_wer = []
difference = []
sentence_lengths = []
werr = []

with open(f'../result/beam_npsc_experiment_27.json', 'r') as file:
    beam_data_points = json.load(file)

for i, wer_point in enumerate(wer_data_points):
    best_option = min(wer_point["wer"])
    best_wer.append(best_option)
    actual_chosen = wer_point["wer_result"]
    actual_wer.append(actual_chosen)
    difference.append(actual_chosen-best_option)
    werr.append(calculate_wer(actual_chosen,best_option))

    if(beam_data_points[i]["audio_file"] == wer_point["audio_file"]):
        sentence_lengths.append(len((beam_data_points[i]["true_transcription"]).split()))

    else:
        sentence_lengths.append(None)



actual_wer_numpy = np.array(actual_wer)
difference_numpy = np.array(difference)
print("sentence_lengths", sentence_lengths)

print("max", max(difference_numpy))

number_chosen_perfect = np.count_nonzero(difference_numpy == 0)

print("number chosen", number_chosen_perfect)
print(number_chosen_perfect/len(difference))

print((mean(actual_wer)-mean(best_wer))/mean(actual_wer))
print(max(sentence_lengths))
print(mean(werr))


plt.figure(figsize=(8, 5))
plt.hist(difference_numpy*100, bins=10, edgecolor="black", alpha=0.7)
plt.xlabel("WER Difference (Model WER - Oracle WER)")
plt.ylabel("Number of Utterances")
plt.title("Histogram of WER Difference")
plt.grid(True)
plt.show()



bins = list(range(0,100, 10)) 
bin_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]

binned_lengths = pd.cut(sentence_lengths, bins=bins, labels=bin_labels)
df = pd.DataFrame({"Sentence Length Bin": binned_lengths, "WER Difference": difference_numpy*100})

plt.figure(figsize=(8, 5))
df.boxplot(column="WER Difference", by="Sentence Length Bin", grid=False)

plt.xlabel("Sentence Length")
plt.ylabel("WER Difference (Model - Oracle)")
plt.title("")
plt.suptitle("")
plt.grid(True)

plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

plt.show()