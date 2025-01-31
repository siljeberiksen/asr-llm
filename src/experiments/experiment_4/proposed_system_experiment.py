import json
import re
from asr.asr_model_initialization import initialize_Whisper_model
from jiwer import wer, cer
import os

from llm.llm import pred


def run_experiment(result_file, beam_file, wer_file, whisper_model):
    true_transcriptions_data = []
    with open('../NPSC/NPSC_1/NPSC_2_0_test.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            # Load each line as a JSON object
            data = json.loads(line)
            true_transcriptions_data.append(data)
    last_element_passed = False
    with open(beam_file, 'r') as file:
            beam_data = json.load(file)
            if beam_data:
                last_element = beam_data[-1]
            else:
                last_element = {"audio_file":None}
                last_element_passed=True

    for true_transcription_data in true_transcriptions_data:
        if(true_transcription_data["sentence_order"])==0:
            context=[]
        if(true_transcription_data['audio'] == last_element["audio_file"] and not last_element_passed):
            last_element_passed=True
            context = last_element["context"]
            continue
        if(not last_element_passed):
            continue

        if (os.path.isfile(os.path.join("../NPSC/NPSC_1", true_transcription_data['audio']))):
            result = whisper_model.transcribe(f"../NPSC/NPSC_1/{true_transcription_data['audio']}", beam_size=5, without_timestamps=True, context=context, integrate_llm=False, experiment_number=4)
        elif (os.path.isfile(os.path.join("../NPSC/NPSC_2", true_transcription_data['audio']))):
            result = whisper_model.transcribe(f"../NPSC/NPSC_2/{true_transcription_data['audio']}", beam_size=5, without_timestamps=True, context=context, integrate_llm=False, experiment_number=4)
        elif (os.path.isfile(os.path.join("../NPSC/NPSC_3", true_transcription_data['audio']))):
            result = whisper_model.transcribe(f"../NPSC/NPSC_3/{true_transcription_data['audio']}", beam_size=5, without_timestamps=True, context=context, integrate_llm=False, experiment_number=4)
        elif  (os.path.isfile(os.path.join("../NPSC/NPSC_4", true_transcription_data['audio']))):
            result = whisper_model.transcribe(f"../NPSC/NPSC_4/{true_transcription_data['audio']}", beam_size=5, without_timestamps=True, context=context, integrate_llm=False, experiment_number=4)
        else:
            result = whisper_model.transcribe(f"../NPSC/NPSC_5/{true_transcription_data['audio']}", beam_size=5, without_timestamps=True, context=context, integrate_llm=False, experiment_number=4)

        print("hallo", result)
#whisper_model = initialize_Whisper_model()   
#run_experiment('../result/npsc_samtale_experiment_4_llm.json', '../result/beam_npsc_experiment_4_llm.json',"../result/wer_npsc_experiment_4_llm.json", whisper_model)

print(pred("Hallo"))
print("hallo")