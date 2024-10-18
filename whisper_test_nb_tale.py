import json
from whisper import load_model
from transformers import WhisperForConditionalGeneration
import re
from jiwer import wer, cer
from pathlib import Path

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
model_name = "NbAiLabBeta/nb-whisper-tiny-verbatim"
#NbAiLabBeta/nb-whisper-small-verbatim
hf_model = WhisperForConditionalGeneration.from_pretrained(model_name)
# Get Hugging Face model's state_dict
hf_state_dict = hf_model.state_dict()

# Convert Hugging Face state dict keys to Whisper's format
whisper_state_dict = {hf_to_whisper_states(k): v for k, v in hf_state_dict.items()}

whisper_model = load_model("tiny") 

# Update Whisper model's state dict with the converted Hugging Face state dict
whisper_model.load_state_dict(whisper_state_dict)


#result = whisper_model.transcribe("NPSC_1/20170216/20170216-095707.wav", beam_size=5, without_timestamps=True)

# path = Path("nb_samtale/data/test/bm/")
# wav_files = [f.name for f in path.glob("*.wav")]

# with open('nb_samtale/metadata.jsonl', 'r') as file:
#     true_transcriptions_data = json.load(file)
true_transcriptions_data = []
with open('nb_samtale/metadata.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # Load each line as a JSON object
        data = json.loads(line)
        true_transcriptions_data.append(data)

for true_transcription_data in true_transcriptions_data:
    result = whisper_model.transcribe(f"nb_samtale/{true_transcription_data['file_name']}", beam_size=5, without_timestamps=True)
    beams_wer=[]
    beams_cer=[]
    beams = []
    with open('result/nb_samtale_without_llm_tiny.json', 'r') as file:
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
    with open("result/wer_nb_samtale_5_tiny.json", 'r') as file:
        wer_data = json.load(file)
    wer_data.append(new_instance)
    with open("result/wer_nb_samtale_5_tiny.json", 'w') as file:
        json.dump(wer_data, file, indent=4)

    new_instance_beams = {
        "sentence_order": true_transcription_data["segment_order"],
        "audio_file": true_transcription_data['file_name'],
        "beams": beams,
        "true_transcription": true_transcription_data['transcription'],
        "transcribed": result["text"]
    }
    with open("result/beams_nb_samtale_5_tiny.json", 'r') as file:
        wer_data = json.load(file)
    wer_data.append(new_instance_beams)
    with open("result/beams_nb_samtale_5_tiny.json", 'w') as file:
        json.dump(wer_data, file, indent=4)