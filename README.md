# ASR-LLM

This repository introduces a system that performs **contextual Beam Selection in Whisper**, leveraging a **LLM-Based Conversational History Integration** to improve ASR accuracy by considering previous segments' context. Developed as part of the course **TDT4501 - Computer Science Immersion Project**.

## Project Structure

The project is organized into Python modules for modularity and clarity, including:

- **asr**: Initializes Whisper, setting up the ASR model for transcription.
- **experiments**: Contains code to run and reproduce experiments, particularly those referenced in the project's documentation.
- **llm**: Manages the calls to the external LLM, hosted on an external server, which provides contextual responses for beam selection.
- **whisper**: A customized copy of the Whisper library, modified to incorporate conversational history into the beam search selection process.

## Running the Project

1. **Set Up the Server**: Ensure that the LLM server is running in the background.
2. **Environment Configuration**: Include the parameters `HOSTNAME` and `PORT` in an `.env` file located in the `src` folder.
3. **Download test data**: In experiments outlined NB Samtale is used - https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-85/

### Running an Experiment

To run Experiment 1, execute the following command from the `src` folder:

```bash
python3 -m experiments.experiment_1.proposed_system_experiment
```
