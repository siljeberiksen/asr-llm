import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def make_violinplot(wer_baseline, wer_baseline_filtered,wer_proposed):
    data = {
        'system': ['baseline'] * len(wer_baseline) + ['baseline'] * len(wer_baseline_filtered) + ['LLM_proposed'] * len(wer_proposed),
        'filtering': ['full'] * len(wer_baseline) + ['filtered'] * len(wer_baseline_filtered) + ['filtered'] * len(wer_baseline_filtered),
        'WER': (
            wer_baseline +       
            wer_baseline_filtered +        
            wer_proposed          
        )
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 6))

    sns.violinplot(data=df, x='filtering', y='WER', hue='system', split=True)

    plt.title('WER Distributions: Baseline vs. LLM (Filtered and Unfiltered)')
    plt.ylabel('WER (%)')
    plt.xlabel('Subset')
    plt.legend(title='System')
    plt.tight_layout()
    plt.show()