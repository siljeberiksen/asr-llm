import json


def checkBestBeamChosen(file_name):
    with open(f'../result/{file_name}', 'r') as file:
        wer_data = json.load(file)
    count_not_best = 0
    count_best = 0
    for data in wer_data:
        beam_wer = data["wer"]
        max_wer = max(beam_wer)
        print(max_wer)
        print(data["wer_result"])
        if(max_wer != data["wer_result"]):
            count_not_best +=1
        else:
            count_best +=1

    print("Count not best:", count_not_best)
    print("Count best", count_best)

checkBestBeamChosen("wer_npsc_experiment_3.json")