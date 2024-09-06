import whisper

model = whisper.load_model("small")

# Load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("resources/test.m4a")
audio = whisper.pad_or_trim(audio)

# Make log-Mel spectrogram and move it to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# Decode the audio using beam search to get multiple possible sentences
options = whisper.DecodingOptions(beam_size=5, without_timestamps=True)  # Adjust beam_size for top sentences
result = whisper.decode(model, mel, options)


##TODO Whisper always returns most likely during beam search, instead want to return all of the most likely

## To test use different temperatures and generate 
# Prepare to store top 5 results
top_sentences = []

# Decode the audio with temperature sampling to get diverse outputs
# for _ in range(5):  # Adjust the range to get 5 results
#     options = whisper.DecodingOptions(temperature=0.7, beam_size=5, ver)
#     result = whisper.decode(model, mel, options)
#     top_sentences.append(result.text)

# Print the top 5 recognized sentences
print("Top 5 possible sentences:")
for i, text in enumerate(top_sentences, start=1):
    print(f"{i}: {text}")

print(result)
