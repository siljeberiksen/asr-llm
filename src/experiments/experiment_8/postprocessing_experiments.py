
import json
import matplotlib.pyplot as plt
import statistics

from visualization.postprocessing_emissions import runEmissionPostProcessing

#with open("result/wer_nb_samtale_5_tiny_2.json", 'r') as file:
#"result/wer_nb_samtale_llm_5_tiny_10_prompt_3.json

def find_empty_instances(file_name):
    with open(f'../result/{file_name}', 'r') as file:
        beam_data = json.load(file)
    empty_instances = []
    for data in beam_data:
        if data["transcribed"] == "":
            empty_instances.append(data["audio_file"])
    return empty_instances

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
    first_started = False
    length = 1
    difference_beams = []
    differnece_beams_cer =[]
    count_files = 0
    for data in wer_data:
        if count_files > 7:
            break
        if(data["sentence_order"]==0):
            count_files +=1
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

    x =  list(range(1, length))
    print("Average wer beam", sum(difference_beams)/length)
    #print(difference_beams)

    print(length)

    print("Count", difference_beams.count(0))

    print("Average cer bemas", sum(differnece_beams_cer)/length)
    print("Count", differnece_beams_cer.count(0))
    median_value = statistics.median(wer_acutal)
    print("median_wer", median_value)

    median_value= statistics.median(cer_actual)
    print("median_cer", median_value)

    print("average wer", wer_sum/length)
    print("average cer", cer_sum/length)

post_process("wer_npsc_experiment_8_llm.json")

print("\n\n\nWithout empty instances")
empty_instances = find_empty_instances("beam_npsc_experiment_8_llm.json")
#post_process("wer_npsc_experiment_3.json", empty_instances)
post_process("wer_npsc_experiment_8_llm.json", empty_instances)

print("\nEmissions")
runEmissionPostProcessing("experiment_8")