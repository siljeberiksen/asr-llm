from transformers import pipeline

# Load the model
asr = pipeline("automatic-speech-recognition", "NbAiLabBeta/nb-whisper-tiny")

#transcribe
# Perform transcription with beam search and multiple return sequences
output = asr("resources/king.mp3", 
             return_timestamps=False,
             generate_kwargs={
                 'num_beams': 10, 
                 'do_sample': True,
                 'task': 'transcribe',
                 'language': 'no',
                 #'num_return_sequences': 10
             })

print(output)
# Print the 5 most likely sentences
for idx, transcription in enumerate(output):
    print(f"Transcription {idx + 1}: {transcription}")