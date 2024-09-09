from whisper import load_model,load_audio,pad_or_trim,log_mel_spectrogram,detect_language,DecodingOptions, decode

model_size = "NbAiLabBeta/nb-whisper-base"
cpu_supported_compute_types = {"small": "int8", "large": "float32"}
precision = "small"
model = load_model("small")

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


result = model.transcribe("data/king.mp3", beam_size=5, without_timestamps=True)
print(result)