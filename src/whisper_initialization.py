import json
from whisper import load_model
from transformers import WhisperForConditionalGeneration
import re
from jiwer import wer, cer

# Found from : https://github.com/openai/whisper/discussions/830
# Used to load a whisper huggingface version into whisper architecture
# This enables huggingface to be loaded into Whisper architecture
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

def initialize_Whisper_model():
    model_name = "NbAiLabBeta/nb-whisper-tiny-verbatim"
    hf_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    hf_state_dict = hf_model.state_dict()
    whisper_state_dict = {hf_to_whisper_states(k): v for k, v in hf_state_dict.items()}
    whisper_model = load_model("tiny") 

    whisper_model.load_state_dict(whisper_state_dict)