# Experiment 43

## Model used

Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf

## Dataset

Utilzing NPSC 3

## Hyperparameters

- beam size = 5
- context size = 10

## Prompt

### Summary

You are a summary creator.
Create a concise summary of the previous sentences generated by an ASR (Automatic Speech Recognition) system.
Focus on the main idea, removing unnecessary filler and side comments.
Preserve essential information such as names, dates, locations, and numerical values.

---

History: [conversational history]

---

Output the summary as plain text.
Always answer in Norwegian

### Prompt for LLM

You are an ASR transcript selector.
Perform language model rescoring based on the top-5 outputs generated by an Automatic Speech Recognition (ASR) system given a conversational history,
Select the most accurate ASR transcription and provide the best possible corrected version if necessary,

---

Summary of conversational history: []

The ASR hypotheses are as follows:
Option 1 [hypothesis 1]
Option 2 [hypothesis 2]
Option 3 [hypothesis 3]
Option 4 [hypothesis 4]
Option 5 [hypothesis 5]

---

Reason concisely about the correct ASR transcription using the summary of past conversational history, and output the best transcription as a plain string. Make minimal corrections if necessary for accuracy and fluency, ensuring it follows the larger conversational history.
Always answer in Norwegian.

## Resulting files

beam_nb_samtale_experiment_43_llm.json
beam_nb_samtale_experiment_3.json

nb_samtale_experiment_43_llm.json
nb_samtale_experiment_3.json

wer_nb_samtale_experiment_43_llm.json
wer_nb_samtale_experiment_3.json

(There were some that was said to be experiment_3, but is actually number 5)

## Schema

class HypothesisSelector(BaseModel):
selected: Literal[1, 2, 3, 4, 5]
