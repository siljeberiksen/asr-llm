import numpy as np
import json
import statistics

def post_process(file_name, empty_instances = []):
    with open(f'../result/{file_name}', 'r') as file:
        wer_data = json.load(file)

    print("Processing: ", file_name)
    wer_acutal =[]
    cer_actual = []
    for data in wer_data:
        if(data["audio_file"] in empty_instances):
            continue

        wer_acutal.append(data["wer_result"])
        cer_actual.append(data["cer_result"])

    return wer_acutal, cer_actual

def standard_deviation(avg_wer_p, avg_wer_b, std_p, std_b, werr):
    print(werr)
    results = abs((1-werr)*(std_b/avg_wer_b-std_p/avg_wer_p))
    return results

def calculate_werr(mean_p, mean_b):
    return (mean_b-mean_p)/mean_b

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
    wer_proposed, cer_proposed = post_process(f'../result/wer_npsc_experiment_{number}_llm.json', files_not_okey)
    wer_baseline, cer_baseline = post_process(f'../result/wer_npsc_experiment_27.json', files_not_okey)
    std_b_wer= statistics.stdev(wer_baseline)
    std_b_cer = statistics.stdev(cer_baseline)
    std_p_cer = statistics.stdev(cer_proposed)
    std_p_wer = statistics.stdev(wer_proposed)
    mean_b_wer= statistics.mean(wer_baseline)
    mean_b_cer = statistics.mean(cer_baseline)
    mean_p_cer = statistics.mean(cer_proposed)
    mean_p_wer = statistics.mean(wer_proposed)

    print("standard diviation wer", standard_deviation(mean_p_wer,mean_b_wer,std_p_wer,std_b_wer, calculate_werr(mean_p_wer,mean_b_wer)))
    print("standard diviation cer", standard_deviation(mean_p_cer,mean_b_cer,std_p_cer,std_b_cer, calculate_werr(mean_p_cer, mean_b_cer)))


understanding_experiment(5)

