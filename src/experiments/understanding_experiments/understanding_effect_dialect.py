import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open(f'../NPSC/NPSC_1/NPSC_2_0_test.jsonl', 'r') as file:
    wer_data_points_df = pd.read_json(file, lines=True)

dialect_files = wer_data_points_df.groupby('speaker_dialect')['audio'].apply(list).to_dict()

norwegian_written_language_files = wer_data_points_df.groupby('sentence_language_code')['audio'].apply(list).to_dict()

gender_filer = wer_data_points_df.groupby('speaker_gender')['audio'].apply(list).to_dict()

def get_wer_lists(file_name, dialect_dict):
    with open(f'../result/{file_name}', 'r') as file:
        wer_data = json.load(file)

    with open(f'../result/wer_npsc_experiment_27.json', 'r') as file:
        wer_data_baseline = json.load(file)

    # Create dict to collect WERs per dialect
    wer_by_dialect = {dialect: [] for dialect in dialect_dict}
    werr_by_dialect = {dialect: [] for dialect in dialect_dict}
    print("Processing: ", file_name)

    for idx, data in enumerate(wer_data):

        audio_file = data.get("audio_file")
        wer = data.get("wer_result")
        wer_baseline = wer_data_baseline[idx].get("wer_result")
        wer_beams = data.get("wer")
        for dialect, audio_list in dialect_dict.items():
            if audio_file in audio_list:
                if(wer_baseline != 0 and wer <= max(wer_beams) and max(wer_beams) != min(wer_beams)):
                    werr_by_dialect[dialect].append(((wer_baseline-wer)/wer_baseline)*100)
                wer_by_dialect[dialect].append(wer)
    return wer_by_dialect, werr_by_dialect

def handle_processing_baseline(experiment, dict_use, filtering_element_str):
    wer_by_element_baseline, werr_by_element_baseline = get_wer_lists(experiment, dict_use)
    
    combined_data_wer = []
    for element_value, werr in wer_by_element_baseline.items():
        combined_data_wer.append({
            filtering_element_str: element_value,
            "wer": werr
        })

    df_wer = pd.DataFrame(combined_data_wer)
    print(df_wer)
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_wer,
        x=filtering_element_str,
        y='wer',
        estimator='mean',
        ci='sd'  # Standard deviation as error bars
    )

def handle_processing(experiment_1, experiment_2, expeirment_3, dict_use, filtering_element_str, x_label):

    wer_by_element_history_10, werr_by_element_history_10 = get_wer_lists(experiment_1, dict_use)
    wer_by_element_history_50, werr_by_element_history_50 = get_wer_lists(experiment_2, dict_use)
    wer_by_element_history_100, werr_by_element_history_100 = get_wer_lists(expeirment_3, dict_use)


    combined_data_werr = []

    for history, wer_dict in zip(
        [10, 50, 100],
        [werr_by_element_history_10, werr_by_element_history_50, werr_by_element_history_100]
    ):
        for filtering_element, werr_list in wer_dict.items():
            for werr in werr_list:
                combined_data_werr.append({
                    filtering_element_str: filtering_element,
                    "werr": werr,
                    "History length": history
                })
    
    # Create DataFrame
    df_werr = pd.DataFrame(combined_data_werr)
    plt.figure(figsize=(10, 6))
    df_werr["History length"] = df_werr["History length"].astype(str)
    ax = sns.barplot(
        data=df_werr,
        x=filtering_element_str,
        y='werr',
        hue='History length',
        estimator='mean',
        ci='sd'  # Standard deviation as error bars
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    plt.xlabel(x_label)
    plt.ylabel("Mean WERR")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# handle_processing("wer_npsc_experiment_12_llm.json","wer_npsc_experiment_18_llm.json","wer_npsc_experiment_17_llm.json", dialect_files, "dialect", "Speaker Dialect")

# handle_processing("wer_npsc_experiment_12_llm.json","wer_npsc_experiment_18_llm.json","wer_npsc_experiment_17_llm.json", norwegian_written_language_files, "language", "Transcription language")

# handle_processing("wer_npsc_experiment_12_llm.json","wer_npsc_experiment_18_llm.json","wer_npsc_experiment_17_llm.json", gender_filer, "gender", "Speaker gender")


# handle_processing("wer_npsc_experiment_14_llm.json","wer_npsc_experiment_20_llm.json","wer_npsc_experiment_21_llm.json", dialect_files, "dialect", "Speaker Dialect")

# handle_processing("wer_npsc_experiment_14_llm.json","wer_npsc_experiment_20_llm.json","wer_npsc_experiment_21_llm.json", norwegian_written_language_files, "language", "Transcription language")

# handle_processing("wer_npsc_experiment_14_llm.json","wer_npsc_experiment_20_llm.json","wer_npsc_experiment_21_llm.json", gender_filer, "gender", "Speaker gender")
handle_processing_baseline("wer_npsc_experiment_27.json", dialect_files, "dialect", )
# handle_processing("wer_npsc_experiment_24_llm.json","wer_npsc_experiment_25_llm.json","wer_npsc_experiment_36_llm.json", dialect_files, "dialect", "Speaker Dialect")

# handle_processing("wer_npsc_experiment_24_llm.json","wer_npsc_experiment_25_llm.json","wer_npsc_experiment_26_llm.json", norwegian_written_language_files, "language", "Transcription language")

# handle_processing("wer_npsc_experiment_24_llm.json","wer_npsc_experiment_25_llm.json","wer_npsc_experiment_36_llm.json", gender_filer, "gender", "Speaker gender")

# wer_by_dialect_history_10, werr_by_dialect_history_10 = get_wer_lists("wer_npsc_experiment_12_llm.json", dialect_files)
# wer_by_dialect_history_50, werr_by_dialect_history_50 = get_wer_lists("wer_npsc_experiment_13_llm.json", dialect_files)
# wer_by_dialect_history_100, werr_by_dialect_history_100 = get_wer_lists("wer_npsc_experiment_14_llm.json", dialect_files)
# # Combine all into one list of dictionaries

# combined_data_werr = []

# for history, wer_dict in zip(
#     [10, 50, 100],
#     [werr_by_dialect_history_10, werr_by_dialect_history_50, werr_by_dialect_history_100]
# ):
#     for dialect, werr_list in wer_dict.items():
#         for werr in werr_list:
#             combined_data_werr.append({
#                 "dialect": dialect,
#                 "werr": werr,
#                 "history": history
#             })

# # Create DataFrame
# df_werr = pd.DataFrame(combined_data_werr)
# plt.figure(figsize=(10, 6))
# df_werr['history'] = df_werr['history'].astype(str)
# ax = sns.barplot(
#     data=df_werr,
#     x='dialect',
#     y='werr',
#     hue='history',
#     estimator='mean',
#     ci='sd'  # Standard deviation as error bars
# )

# for container in ax.containers:
#     ax.bar_label(container, fmt='%.2f', padding=3)

# plt.title("Mean WERR per Dialect by History Length (with Std Dev)")
# plt.xlabel("Speaker Dialect")
# plt.ylabel("Mean WERR")
# plt.grid(True, axis='y')
# plt.tight_layout()
# plt.show()