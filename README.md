# ASR-LLM

This repository introduces a system, Whisper-GEC-H, that performs **contextual Beam Selection in Whisper**, leveraging a **LLM-Based Conversational History Integration** to improve ASR accuracy by considering previous segments' context. Developed as part of the course **TDT4900 - Computer Science, Master's Thesis**.

## Project Structure

The project is organised into Python modules for modularity and clarity, including:

- **asr**: Initialises Whisper, setting up the ASR model for transcription.
- **experiments**: Contains code to run and reproduce experiments, particularly those referenced in the project's documentation.
- **llm**: Manages the calls to the external LLM, hosted on an external server, which provides contextual responses for beam selection.
- **whisper**: A customised copy of the Whisper library, modified to incorporate conversational history into the beam search selection process.

## Whisper-GEC-H

## Development of Whisper-GEC-H

The open-source Whisper Python library was extended in this project to support the **Whisper-GEC-H** framework. This extension introduces four optional arguments to the `transcribe()` method, allowing integration with an external LLM and conversational history tracking:

- context (Optional[str]):
 An optional list of strings containing the conversational history so far. This is added to the prompt to help the LLM handle context-dependent utterances. 

- integrate_llm (Optional[bool]):
When set to True, it enables LLM integration, allowing the external LLM to select the most likely transcription from beam search candidates (as defined in the GEC-H framework).

- port (Optional[int], default=8081):
Specifies the port used to communicate with the external LLM server. The default is 8081.

- experiment_number (Optional[int]):
An optional identifier used to save or log context related to a specific experiment. Used for experimental purposes, to save the output given by the LLM

The Whisper-GEC-H beam search decoding needs to be used, as this is where the external LLM is used. An example call to the new Whisper module can therefore be.:
````python
Whisper.transcribe(audio_path, beam_size=5, context=[], integrate_llm=True, port=8081, experiment_number=3)
````

## Coding pipeline


## Running the Project

1. **Set Up the Server**: Ensure that the LLM server is running in the background.
2. **Download test data**: In experiments outlined NB Samtale is used - https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-85/

### Running an Experiment

To run Experiment 1, execute the following command from the `src` folder:

```bash
python3 -m experiments.experiment_1.proposed_system_experiment
````

### Running an experiment on a server

As the experiments do not require supervision, a way to make the exeriment run even when not using own computer is to run it from a server.
Steps to doing this:

1. Clone the git project to the server

2. Download all training files intp NPSC folder

3. Run commands using tmux, nohup or window to let the program run even when current bash session is closed

