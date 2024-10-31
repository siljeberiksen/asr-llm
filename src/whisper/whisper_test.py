import json
from whisper import load_model
from transformers import WhisperForConditionalGeneration
import re
from jiwer import wer, cer

# Found from 
# https://github.com/openai/whisper/discussions/830
def hf_to_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    text = re.sub('proj_out.weight', 'decoder.token_embedding.weight', text)
    return text
# Load Hugging Face model
model_name = "NbAiLabBeta/nb-whisper-small-verbatim"
#NbAiLabBeta/nb-whisper-small-verbatim
hf_model = WhisperForConditionalGeneration.from_pretrained(model_name)
# Get Hugging Face model's state_dict
hf_state_dict = hf_model.state_dict()

# Convert Hugging Face state dict keys to Whisper's format
whisper_state_dict = {hf_to_whisper_states(k): v for k, v in hf_state_dict.items()}

whisper_model = load_model("small") 

# Update Whisper model's state dict with the converted Hugging Face state dict
whisper_model.load_state_dict(whisper_state_dict)


#result = whisper_model.transcribe("NPSC_1/20170216/20170216-095707.wav", beam_size=5, without_timestamps=True)


# Want to get WER (word error rate) on the transcriptions from 20170207 to see wether there are any reasons for doing it. 
with open('NPSC_1/20170207/20170207_sentence_data.json', 'r') as file:
    true_transcription = json.load(file)
#Get the output from the beam search

for transcribed_sentence in true_transcription["sentences"]:
    result = whisper_model.transcribe(f"NPSC_1/20170207/audio/{transcribed_sentence['audio_file']}", beam_size=5, without_timestamps=True)
    beams_wer=[]
    beams_cer=[]
    beams = []
    with open('result/nb_samtale_without_llm.json', 'r') as file:
        beam_options = json.load(file)[-1]["choices"]
        for beam_option in beam_options:
            beams_wer.append(wer(transcribed_sentence["nonverbatim_text"].lower(), beam_option.lower()))
            beams_cer.append(cer(transcribed_sentence["nonverbatim_text"].lower(), beam_option.lower()))
            beams.append(beam_option.lower())
    new_instance = {
            "sentence_order": transcribed_sentence["sentence_order"],
            "audio_file": transcribed_sentence['audio_file'],
            "wer": beams_wer,
            "cer": beams_cer
        }
    with open("result/wer_test.json", 'r') as file:
        wer_data = json.load(file)
    wer_data.append(new_instance)
    with open("result/wer_test.json", 'w') as file:
        json.dump(wer_data, file, indent=4)

    new_instance_beams = {
        "sentence_order": transcribed_sentence["sentence_order"],
        "audio_file": transcribed_sentence['audio_file'],
        "beams": beams,
        "true_transcription": transcribed_sentence["nonverbatim_text"]
    }
    with open("result/beams_test.json", 'r') as file:
        wer_data = json.load(file)
    wer_data.append(new_instance_beams)
    with open("result/beams_test.json", 'w') as file:
        json.dump(wer_data, file, indent=4)
