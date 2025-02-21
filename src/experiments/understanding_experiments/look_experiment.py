import json
import statistics
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


def understanding_experiment(number):
    with open(f'../result/wer_npsc_experiment_{number}_llm.json', 'r') as file:
        wer_data_points = json.load(file)
    files = []
    files_not_okey = []
    for wer_data in wer_data_points:
        beam_wer = wer_data["wer"]
        max_wer = max(beam_wer)
        if(max_wer >= wer_data["wer_result"] and max_wer-min(beam_wer)!=0):
            files.append(wer_data["audio_file"])
        else:
            files_not_okey.append(wer_data["audio_file"])
    print("Results better", len(files_not_okey))
    print("Results total", len(wer_data_points))
    data_points_to_add =[]

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
    post_process(f'../result/wer_npsc_experiment_{number}_llm.json', files_not_okey)
    print("Basline")
    post_process(f'../result/wer_npsc_experiment_3.json', files_not_okey)


understanding_experiment(7)