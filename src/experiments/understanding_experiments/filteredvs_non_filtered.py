import json
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from visualization.creating_distribution_of_wer import creating_distribution
def post_process(file_name, empty_instances = []):
    with open(f'../result/{file_name}', 'r') as file:
        wer_data = json.load(file)

    print("Processing: ", file_name)
    wer_best_beam = []
    cer_best_beam = []
    wer_worst_beam = []
    cer_worst_beam = []
    wer_sum = 0
    cer_sum = 0
    wer_acutal =[]
    cer_actual = []
    length = 0
    difference_beams = []
    differnece_beams_cer =[]
    for data in wer_data:
        if(data["audio_file"] in empty_instances):
            continue
        cer_best_beam.append(min(data["cer"]))
        wer_worst_beam.append(max(data["wer"]))
        cer_worst_beam.append(max(data["cer"]))
        difference_beams.append(max(data["wer"])-min(data["wer"]))
        differnece_beams_cer.append(max(data["cer"])-min(data["cer"]))

        wer_acutal.append(data["wer_result"])
        cer_actual.append(data["cer_result"])

        wer_sum += data["wer_result"]
        cer_sum += data["cer_result"]
        length +=1
    print("Average wer beam", sum(difference_beams)/length)


    print(length)

    print("Count", difference_beams.count(0))

    print("Average cer bemas", sum(differnece_beams_cer)/length)
    print("Count", differnece_beams_cer.count(0))
    median_value = statistics.median(wer_acutal)
    print("median_wer", median_value)

    median_value= statistics.median(cer_actual)
    print("median_cer", median_value)

    print("average wer", wer_sum/length)
    print("Standard deviation wer", statistics.stdev(wer_acutal))
    print("average cer", cer_sum/length)
    print("Standard devision cer", statistics.stdev(cer_actual))
    return  wer_sum/length, wer_acutal

def understanding_experiment(number):
    with open(f'../result/wer_npsc_experiment_{number}_llm.json', 'r') as file:
        wer_data_points = json.load(file)
    files = []
    files_not_okey = []
    wer_list = []
    for wer_data in wer_data_points:
        beam_wer = wer_data["wer"]
        max_wer = max(beam_wer)
        if(max_wer >= wer_data["wer_result"] and max_wer !=min(beam_wer)):
            files.append(wer_data["audio_file"])
            wer_list.append(wer_data["wer_result"]*100)
        else:
            files_not_okey.append(wer_data["audio_file"])
    print("Results better", len(files_not_okey))
    print("Results total", len(wer_data_points))
    data_points_to_add =[]

    #creating_distribution(wer_list)

    with open(f'../result/beam_npsc_experiment_{number}_llm.json', 'r') as file:
        beam_data_points = json.load(file)
        for beam_data in beam_data_points:
            if(beam_data["audio_file"] in files_not_okey):
                data_point_to_add = {
                    "beams": beam_data["beams"] ,
                    "transcribed": beam_data["transcribed"],
                    "audio_file": beam_data["audio_file"],
                }
                data_points_to_add.append(data_point_to_add)
        print(len(data_points_to_add))

    with open(f'../result/filtering_empty_experiment_{number}_llm.json', "w+") as file:
        file.write(json.dumps(data_points_to_add, indent=4))
    mean_filter, wer_values = post_process(f'../result/wer_npsc_experiment_{number}_llm.json', files_not_okey)
    mean_non_filter, wer_values_filter = post_process(f'../result/wer_npsc_experiment_{number}_llm.json')
    print("Basline")
    mean_filter_basleine, wer_values_filter_baseline = post_process(f'../result/wer_npsc_experiment_27.json', files_not_okey)
    return mean_filter, wer_values, mean_non_filter, wer_values_filter, mean_filter_basleine, wer_values_filter_baseline

mean_filter_10, wer_values_llm_full_10, mean_non_filter_10, wer_values_llm_filtered_10, mean_filter_baseline_10, wer_values_baseline_filtered_10 = understanding_experiment(7)

mean_filter_50, wer_values_llm_full_50, mean_non_filter_50, wer_values_llm_filtered_50, mean_filter_baseline_50, wer_values_baseline_filtered_50 = understanding_experiment(6)

mean_filter_100, wer_values_llm_full_100, mean_non_filter_100, wer_values_llm_filtered_100, mean_filter_baseline_100, wer_values_baseline_filtered_100 = understanding_experiment(5)



mean_filter_10_llama, wer_values_llm_full_10_llama, mean_non_filter_10, wer_values_llm_filtered_10, mean_filter_baseline_10_llama, wer_values_baseline_filtered_10 = understanding_experiment(9)
mean_filter_50_llama, wer_values_llm_full_50_llama, mean_non_filter_50, wer_values_llm_filtered_50, mean_filter_baseline_50_llama, wer_values_baseline_filtered_50 = understanding_experiment(10)

mean_filter_100_llama, wer_values_llm_full_100_llama, mean_non_filter_100, wer_values_llm_filtered_100, mean_filter_baseline_100_llama, wer_values_baseline_filtered_100 = understanding_experiment(11)

mean_baseline, wer_values_baseline = post_process(f'../result/wer_npsc_experiment_27.json')

mean_filter_10
mean_filter_50
mean_filter_100

mean_df = pd.DataFrame({
    'History': [10, 10, 50, 50, 100, 100],
    'Filtering based on LLM': ['Baseline filtering Gemma', 'Baseline filtering LLaMA'] * 3,
    'Mean_WER': [
        mean_filter_baseline_10*100, mean_filter_baseline_10_llama*100,
        mean_filter_baseline_50*100, mean_filter_baseline_50_llama*100,
        mean_filter_baseline_100*100, mean_filter_baseline_100_llama*100,
    ]
})
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=mean_df, x='History', y='Mean_WER', hue='Filtering based on LLM')
ax.axhline(
    y=mean_baseline * 100,
    color='black',
    linestyle='dashed',
    linewidth=1.2,
    label='Baseline non filtering'
)
print(mean_filter_baseline_10)
print(mean_filter_baseline_50)
print(mean_filter_baseline_100)

for container in ax.containers:
    ax.bar_label(container, fmt='%.4f%%', padding=3)

ax.yaxis.grid(True, linestyle='--', linewidth=0.7)
handles, labels = ax.get_legend_handles_labels()

unique = dict(zip(labels, handles))

ax.legend(unique.values(), unique.keys(), title='')
ax.set_axisbelow(True) 

plt.title('')
plt.ylabel('Mean WER(%)')
plt.xlabel('History Length')
plt.ylim(0, max(mean_df['Mean_WER'])+5) 
plt.tight_layout()
plt.show()