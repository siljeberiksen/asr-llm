import json
import re
from asr.asr_model_initialization import initialize_Whisper_model
from jiwer import wer, cer


def run_experiment(result_file, beam_file, wer_file, whisper_model):
    true_transcriptions_data = []
    with open('../nb_samtale/metadata.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            # Load each line as a JSON object
            data = json.loads(line)
            true_transcriptions_data.append(data)

    last_element_passed = False
    for true_transcription_data in true_transcriptions_data:
        if(true_transcription_data["segment_order"])==0:
            context=[]
        with open(beam_file, 'r') as file:
            last_element = json.load(file)[-1]
        if(true_transcription_data['file_name'] == last_element["audio_file"] and not last_element_passed):
            last_element_passed=True
            context = last_element["context"]
            continue
        if(not last_element_passed):
            continue
        result = whisper_model.transcribe(f"../nb_samtale/{true_transcription_data['file_name']}", beam_size=5, without_timestamps=True, context=context, integrate_llm=True)
            
        beams_wer=[]
        beams_cer=[]
        beams = []
        with open(result_file, 'r') as file:
            beam_options = json.load(file)[-1]["choices"]
            for beam_option in beam_options:
                beams_wer.append(wer(true_transcription_data["transcription"].lower(), beam_option.lower()))
                beams_cer.append(cer(true_transcription_data["transcription"].lower(), beam_option.lower()))
                beams.append(beam_option.lower())
        new_instance = {
                "sentence_order": true_transcription_data["segment_order"],
                "audio_file": true_transcription_data['file_name'],
                "wer": beams_wer,
                "cer": beams_cer,
                "wer_result": wer(true_transcription_data["transcription"].lower(), result["text"].lower()),
                "cer_result": cer(true_transcription_data["transcription"].lower(), result["text"].lower())
            }
        with open(wer_file, 'r') as file:
            wer_data = json.load(file)
        wer_data.append(new_instance)
        with open(wer_file, 'w') as file:
            json.dump(wer_data, file, indent=4)

        new_instance_beams = {
            "sentence_order": true_transcription_data["segment_order"],
            "audio_file": true_transcription_data['file_name'],
            "beams": beams,
            "true_transcription": true_transcription_data['transcription'],
            "transcribed": result["text"],
            "context": context
        }
        with open(beam_file, 'r') as file:
            wer_data = json.load(file)
        wer_data.append(new_instance_beams)
        with open(beam_file, 'w') as file:
            json.dump(wer_data, file, indent=4)
        if(len(context) >= 10):
            context.pop(0)
        context.append(result["text"])

whisper_model = initialize_Whisper_model()   
run_experiment('../result/nb_samtale_experiment_1_llm.json', '../result/beam_nb_samtale_experiment_1_llm.json',"../result/wer_nb_samtale_experiment_1_llm.json", whisper_model)