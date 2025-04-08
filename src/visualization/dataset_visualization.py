import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter



true_transcriptions_data = []
with open('../NPSC/NPSC_1/NPSC_2_0_test.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        # Load each line as a JSON object
        data = json.loads(line)
        true_transcriptions_data.append(data)
word_lengths = []
speaker_counter = Counter()
dialect_counter = Counter()
gender_counter = Counter()
language_counter = Counter()
for true_transcription_data in true_transcriptions_data:
    # Get transcription text
    transcription_text = true_transcription_data.get('nonverbatim_text', None)
    if transcription_text is not None:
        num_words = len(transcription_text.split())
        word_lengths.append(num_words)
    
    # Get speaker name
    transcription_name = true_transcription_data.get('speaker_name', None)
    if transcription_name:
        speaker_counter[transcription_name] += 1
    
    # Get speaker dialect
    transcription_dialect = true_transcription_data.get('speaker_dialect', None)
    if transcription_dialect:
        dialect_counter[transcription_dialect] += 1
    
    # Get speaker geneder
    transcription_gender = true_transcription_data.get('speaker_gender', None)
    if transcription_gender:
        gender_counter[transcription_gender] += 1

    # Get language
    transcription_language = true_transcription_data.get('sentence_language_code', None)
    if transcription_language:
        language_counter[transcription_language] += 1


# Remove all zero-length transcriptions
word_lengths = [w for w in word_lengths if w > 0]
# Now plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(word_lengths, bins=30, edgecolor='black', log=True)
plt.xlabel('Number of Words')
plt.ylabel('Number of Segments')
plt.title('')
plt.grid(True)
plt.xlim(0, None)

plt.show()

# Print basic statistics
print(f"Number of segments: {len(word_lengths)}")
print(f"Number of unique speakers: {len(speaker_counter)}")
print(f"Number of unique dialects: {len(dialect_counter)}")
print(f"Number of unique dialects: {len(gender_counter)}")
print(f"Number of unique langauge: {len(language_counter)}")

# Show top 10 speakers and dialects
print("\nTop 10 speakers by number of segments:")
for speaker, count in speaker_counter.most_common(10):
    print(f"{speaker}: {count}")

print("\nTop 10 dialects by number of segments:")
for dialect, count in dialect_counter.most_common(10):
    print(f"{dialect}: {count}")

print("\nTop 2 gender by number of segments:")
for gender, count in gender_counter.most_common(2):
    print(f"{gender}: {count}")
percentage_male = (3822 / 6355) * 100
percentage_female = (2533 / 6355) * 100
print(percentage_male)
print(percentage_female)

print("\nTop 2 language by number of segments:")
for language, count in language_counter.most_common(2):
    print(f"{language}: {count}")

percentage_BB = (5527 / (5527+828)) * 100
percentage_nn = (827 / (5527+828)) * 100
print(percentage_BB)
print(percentage_nn)



print("Mean number of words:", np.mean(word_lengths))
print("Median number of words:", np.median(word_lengths))
print("Standard deviation:", np.std(word_lengths))
dialects, counts = zip(*dialect_counter.most_common())
plt.figure(figsize=(8, 6))
#plt.bar(dialects, counts)
plt.pie(counts, labels=dialects, autopct='%1.1f%%', startangle=140)
#plt.xticks(rotation=0)
# plt.xlabel('Dialect')
# plt.ylabel('Number of Segments')
plt.title('')
plt.grid(True)
plt.show()