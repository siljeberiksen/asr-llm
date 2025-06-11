import json
import numpy as np
import matplotlib.pyplot as plt
with open(f'../result/wer_npsc_experiment_19_llm.json', 'r') as file:
        wer_datas = json.load(file)

wer_values = []

for wer_data in wer_datas:
    wer_values.append(wer_data["wer_result"]*100)

def creating_distribution(wer_values):
    print(max(wer_values))
    bins = np.arange(0, max(wer_values) + 4000, 4000)  # Buckets in steps of 20

    plt.figure(figsize=(10, 6))
    plt.hist(wer_values, bins=bins, edgecolor='black', alpha=0.75)
    plt.yscale("log")

    plt.xlabel("WER (%)", fontsize=40)
    plt.ylabel("Frequency", fontsize=40)
    plt.title("")
    plt.xticks(bins[::2],fontsize=40)  # Ensure all bins are labeled
    plt.yticks(fontsize=40)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


creating_distribution(wer_values)