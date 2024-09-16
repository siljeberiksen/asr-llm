from whisper import load_model,load_audio,pad_or_trim,log_mel_spectrogram,detect_language,DecodingOptions, decode
from transformers import WhisperForConditionalGeneration
import re

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


# This is code to initate only one transcribtion
# # Load audio and pad/trim it to fit 30 seconds
# audio = load_audio("data/king.mp3")
# audio = pad_or_trim(audio)

# # Make log-Mel spectrogram and move it to the same device as the model
# mel = log_mel_spectrogram(audio).to(model.device)

# # Detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# Decode the audio using beam search to get multiple possible sentences
# options = DecodingOptions(beam_size=5, without_timestamps=True)  # Adjust beam_size for top sentences
# result = decode(model, mel, options)


result = whisper_model.transcribe("data/king.mp3", beam_size=5, without_timestamps=True)
print("result", result)