# Experiment 26

## Model used

Just running baseline

## Resulting files

beam_nb_samtale_experiment_26_llm.json
beam_nb_samtale_experiment_3.json

nb_samtale_experiment_26_llm.json
nb_samtale_experiment_3.json

wer_nb_samtale_experiment_26_llm.json
wer_nb_samtale_experiment_3.json

(There were some that was said to be experiment_3, but is actually number 5)

## Schema

class HypothesisSelector(BaseModel):
selected: Literal[1, 2, 3, 4, 5]
