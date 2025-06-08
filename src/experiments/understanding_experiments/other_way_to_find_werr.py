import json
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
def calculate_wer(baseline, proposed):
    if(baseline ==0):
        return 0
    else:
        return (baseline-proposed)/baseline 
    
def find_std_analysis(file_name, baseline_file, files_not_okey=[]):
    with open(f'../result/{file_name}', 'r') as file:
        wer_data = json.load(file)
    with open(f'../result/{baseline_file}', 'r') as file:
        wer_data_baseline = json.load(file)
    true_transcriptions = []
    with open("../NPSC/NPSC_1/NPSC_2_0_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            true_transcriptions.append(json.loads(line))
    # with open(f'../result/{baseline_file}', 'r') as file:
    #     wer_data_baseline = json.load(file)
    werr_points= []
    cerr_points= []
    files_with_positive_wer_negative_cer =[]
    for i in range(0,len(wer_data)):
        if(wer_data[i]["audio_file"] in files_not_okey):
            continue
        if(len(true_transcriptions[i]["nonverbatim_text"].split())<40):
            print(true_transcriptions[i]["nonverbatim_text"].split())
            print("continoue")
            continue
        werr_point=calculate_wer(wer_data_baseline[i]["wer_result"],wer_data[i]["wer_result"])
        cerr_point= calculate_wer(wer_data_baseline[i]["cer_result"],wer_data[i]["cer_result"])
        werr_points.append(werr_point)
        cerr_points.append(cerr_point)
        # TODO used when finding instance 
        # if(werr_point>0 and cerr_point<0):
        #     print("jaaaaaaaa")
        #     print(wer_data_baseline[i]["audio_file"])
        #     files_with_positive_wer_negative_cer.append(wer_data_baseline[i]["audio_file"])
        #     print(wer_data[i]["wer_result"])
        #     print(wer_data_baseline[i]["wer_result"])
        #     print(wer_data[i]["cer_result"])
        #     print(wer_data_baseline[i]["cer_result"])
        #     print(werr_point)
        #     print(cerr_point)

    # with open(f'../result/beam_npsc_experiment_20_llm.json', 'r') as file:
    #     beam_data_points = json.load(file)
    # with open(f'../result/beam_npsc_experiment_27.json', 'r') as file:
    #     beam_data_points_baseline = json.load(file)
    # for i in range(0,len(wer_data)):
    #     if(beam_data_points[i]["audio_file"] in files_with_positive_wer_negative_cer):
    #         print(beam_data_points[i]["audio_file"])
    #         print(beam_data_points[i]["transcribed"])
    #         print(beam_data_points_baseline[i]["transcribed"])
    #         print("\n\n\n\n")
    
    mean_wer = np.mean(werr_points)
    mean_cer = np.mean(cerr_points)

    median_wer = np.median(werr_points)
    median_cer = np.median(cerr_points)
    standard_deviation_wer = np.std(werr_points,ddof=0)
    standard_deviation_cer=  np.std(cerr_points,ddof=0)

    print(mean_wer*100)
    print(standard_deviation_wer*100)

    print(mean_cer*100)
    print(standard_deviation_cer*100)

    print(median_wer*100)
    print(median_cer*100)
    return mean_wer*100, standard_deviation_wer*100

def find_std(file_name, baseline_file, files_not_okey=[]):
    with open(f'../result/{file_name}', 'r') as file:
        wer_data = json.load(file)
    with open(f'../result/{baseline_file}', 'r') as file:
        wer_data_baseline = json.load(file)
    # with open(f'../result/{baseline_file}', 'r') as file:
    #     wer_data_baseline = json.load(file)
    werr_points= []
    cerr_points= []
    for i in range(0,len(wer_data)):
        if(wer_data[i]["audio_file"] in files_not_okey):
            continue
        werr_points.append(calculate_wer(wer_data_baseline[i]["wer_result"],wer_data[i]["wer_result"]))
        cerr_points.append(calculate_wer(wer_data_baseline[i]["cer_result"],wer_data[i]["cer_result"]))
    mean_wer = np.mean(werr_points)
    mean_cer = np.mean(cerr_points)

    median_wer = np.median(werr_points)
    median_cer = np.median(cerr_points)
    standard_deviation_wer = np.std(werr_points,ddof=0)
    standard_deviation_cer=  np.std(cerr_points,ddof=0)

    print(mean_wer*100)
    print(standard_deviation_wer*100)

    print(mean_cer*100)
    print(standard_deviation_cer*100)

    print(median_wer*100)
    print(median_cer*100)
    return mean_wer*100, standard_deviation_wer*100

def understanding_experiment(number, baseline_number):
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
    return find_std(f'wer_npsc_experiment_{number}_llm.json', f'wer_npsc_experiment_{baseline_number}.json', files_not_okey)
    

def create_line_chart():
    mean_prompt_1 =[]
    standard_deviation_prompt_1 = []

    result_10 = understanding_experiment(12,27)
    result_50 = understanding_experiment(18,27)
    result_100 = understanding_experiment(17,27)
    print(result_10)

    mean_prompt_1.append(result_10[0])
    mean_prompt_1.append(result_50[0])
    mean_prompt_1.append(result_100[0])

    standard_deviation_prompt_1.append(result_10[1])
    standard_deviation_prompt_1.append(result_50[1])
    standard_deviation_prompt_1.append(result_100[1])

    mean_prompt_2 = []
    standard_deviation_prompt_2 = []


    result_10 = understanding_experiment(14,27)
    result_50 = understanding_experiment(20,27)
    result_100 = understanding_experiment(21,27)

    mean_prompt_2.append(result_10[0])
    mean_prompt_2.append(result_50[0])
    mean_prompt_2.append(result_100[0])

    standard_deviation_prompt_2.append(result_10[1])
    standard_deviation_prompt_2.append(result_50[1])
    standard_deviation_prompt_2.append(result_100[1])

    print(mean_prompt_1)
    print(standard_deviation_prompt_1)

    x_labels = ["10", "50", "100"]  # Categorical x-axis labels



    positions = range(len(x_labels))

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(positions, mean_prompt_1, label="Prompt 1", marker='o', color='tab:green')  # WER line
    plt.plot(positions, mean_prompt_2, label="Prompt 2", marker='o', color='tab:blue')  # WER line
    # plt.plot(positions, CER_MEDIAN, label="CERR mean", marker='s',  color='tab:blue', linestyle='-')  # CER line
    # plt.plot(positions, WER, label="WERR median", marker='o', color='tab:orange', linestyle='--')  # WER line
    # plt.plot(positions, CER, label="CERR median", marker='s', color='tab:orange', linestyle='-')  # CER line
    plt.xticks(ticks=positions, labels=x_labels)

    # Set y-axis limits from 0 to 100
    #

    # plt.errorbar(x_labels,mean_prompt_1, yerr=standard_deviation_prompt_1, fmt="o", color="r")
    # plt.errorbar(x_labels,mean_prompt_2, yerr=standard_deviation_prompt_2, fmt="o", color="r")
    # Add titles and labels
    plt.title("")
    plt.xlabel("History Length (l)")
    plt.ylabel("WERR (%)")

    # # Format y-axis labels to show percentages
    # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

find_std_analysis("wer_npsc_experiment_23_llm.json", "wer_npsc_experiment_27.json")
#create_line_chart()
#create_line_chart()
# create_line_chart()