import json
import numpy as np
import matplotlib.pyplot as plt
with open(f'../result/wer_npsc_experiment_30_llm.json', 'r') as file:
        wer_datas = json.load(file)

wer_values = []

for wer_data in wer_datas:
    wer_values.append(wer_data["wer_result"]*100)
# Example WER values (some exceeding 100%)
#wer_values = [5, 12, 18, 25, 30, 42, 55, 60, 75, 90, 110, 120, 135, 150]

# Define bins (including values >100%)

def creating_distribution(wer_values):
    print(max(wer_values))
    bins = np.arange(0, max(wer_values) + 500, 500)  # Buckets in steps of 20

    # Create histogram
    plt.figure(figsize=(8, 5))
    plt.hist(wer_values, bins=bins, edgecolor='black', alpha=0.75)

    # Labels and title
    plt.xlabel("WER (%)")
    plt.ylabel("Frequency")
    plt.title("")
    plt.xticks(bins)  # Ensure all bins are labeled
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()


creating_distribution(wer_values)