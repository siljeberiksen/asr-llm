import json
import re
from asr.asr_model_initialization import initialize_Whisper_model
from jiwer import wer, cer
import os


def run_experiment(result_file, beam_file, wer_file, whisper_model):
    true_transcriptions_data = []
    with open('../NPSC/NPSC_1/NPSC_2_0_test.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            # Load each line as a JSON object
            data = json.loads(line)
            true_transcriptions_data.append(data)
    print(true_transcriptions_data)
    last_element_passed = False
    with open(beam_file, 'r') as file:
            if json.load(file):
                last_element = json.load(file)[-1]
            else:
                last_element = {"audio_file":None}
                last_element_passed=True
    for true_transcription_data in true_transcriptions_data:
        if(true_transcription_data['audio'] == last_element["audio_file"] and not last_element_passed):
            last_element_passed=True
            continue
        if(not last_element_passed):
            continue

        if (os.path.isfile(os.path.join("../NPSC/NPSC_1", true_transcription_data['audio']))):
             result = whisper_model.transcribe(f"../NPSC/NPSC_1/{true_transcription_data['audio']}", beam_size=5, without_timestamps=True)
        elif (os.path.isfile(os.path.join("../NPSC/NPSC_2", true_transcription_data['audio']))):
            result = whisper_model.transcribe(f"../NPSC/NPSC_2/{true_transcription_data['audio']}", beam_size=5, without_timestamps=True)
        elif (os.path.isfile(os.path.join("../NPSC/NPSC_3", true_transcription_data['audio']))):
             result = whisper_model.transcribe(f"../NPSC/NPSC_3/{true_transcription_data['audio']}", beam_size=5, without_timestamps=True)
        else:
             result = whisper_model.transcribe(f"../NPSC/NPSC_4/{true_transcription_data['audio']}", beam_size=5, without_timestamps=True)
       
            
        beams_wer=[]
        beams_cer=[]
        beams = []
        with open(result_file, 'r') as file:
            beam_options = json.load(file)[-1]["choices"]
            for beam_option in beam_options:
                beams_wer.append(wer(true_transcription_data["nonverbatim_text"].lower(), beam_option.lower()))
                beams_cer.append(cer(true_transcription_data["nonverbatim_text"].lower(), beam_option.lower()))
                beams.append(beam_option.lower())
        new_instance = {
                "sentence_order": true_transcription_data["sentence_order"],
                "audio_file": true_transcription_data['audio'],
                "wer": beams_wer,
                "cer": beams_cer,
                "wer_result": wer(true_transcription_data["nonverbatim_text"].lower(), result["text"].lower()),
                "cer_result": cer(true_transcription_data["nonverbatim_text"].lower(), result["text"].lower())
            }
        with open(wer_file, 'r') as file:
            wer_data = json.load(file)
        wer_data.append(new_instance)
        with open(wer_file, 'w') as file:
            json.dump(wer_data, file, indent=4)

        new_instance_beams = {
            "sentence_order": true_transcription_data["sentence_order"],
            "audio_file": true_transcription_data["audio"],
            "beams": beams,
            "true_transcription": true_transcription_data['nonverbatim_text'],
            "transcribed": result["text"]
        }
        with open(beam_file, 'r') as file:
            wer_data = json.load(file)
        wer_data.append(new_instance_beams)
        with open(beam_file, 'w') as file:
            json.dump(wer_data, file, indent=4)
whisper_model = initialize_Whisper_model()   
run_experiment('../result/npsc_samtale_experiment_2.json', '../result/beam_nb_samtale_experiment_2.json',"../result/wer_nb_samtale_experiment_2.json", whisper_model)