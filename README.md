# ASR-LLM

This repository introduces Whisper-GEC-H, a system that performs contextual Beam Selection in Whisper, leveraging an LLM-Based Conversational History Integration to improve ASR accuracy by considering previous segments' context. Whisper-GEC-H is developed as part of the course TDT4900 Computer Science, Master's Thesis.

## Project Structure

The project is organised into Python modules for modularity and clarity, including:

- **asr**: Initialises Whisper, setting up the ASR model for transcription.
- **experiments**: Contains code to run and reproduce experiments, particularly those referenced in the project's documentation.
- **llm**: Manages the calls to the external LLM, hosted on an external server, which provides contextual responses for beam selection.
- **whisper**: A customised copy of the Whisper library, modified to incorporate conversational history into the beam search selection process.

## Whisper-GEC-H
The Whisper-GEC-H system is built on the GEC-H framework, which can be applied to any ASR model. It employs beam search decoding, but instead of selecting the most probable hypothesis based on conventional scoring, it sends the beam candidates along with the previous transcriptions to a large language model (LLM). The LLM selects the most appropriate transcription, which is then used to update the sequence of prior transcriptions. 

The framework can be visualised as:
![proposed_GEC_pipeline drawio (5) (1)-1](https://github.com/user-attachments/assets/22d9270c-52c2-4498-be88-7730e1ee6942)

Whisper-GEC-H builds on the Whisper model, extending its functionality to operate within the GEC-H framework. The system can be visualized as follows:
![proposed drawio (6) (1)-1](https://github.com/user-attachments/assets/c3d41387-d11e-4d0c-adb4-12f737421b48)


## Development of Whisper-GEC-H

The open-source Whisper Python library was extended in this project to support the **Whisper-GEC-H** framework. This extension introduces four optional arguments to the `transcribe()` method, allowing integration with an external LLM and conversational history tracking:

- context (Optional[str[]):
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

## Running the Project

1. **Set Up the Server**: Ensure the LLM server runs in the background. This can, for example, be done using ollama. If this set-up is changed, the module titled llm must be updated accordingly. 
2. **Download test data**: Download [NPSC 2.0](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-85/) into the ASR-LLM folder, and title the file NPSC

### Installing and Running Ollama

To install Ollama on a Linux system:

```bash
# Download and extract Ollama
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
mkdir -p ~/.local
tar -C ~/.local -xzf ollama-linux-amd64.tgz

# Add Ollama to PATH and library path (temporary for this session)
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH

# Persist settings in ~/.bashrc
grep -qxF 'export PATH=$HOME/.local/bin:$PATH' ~/.bashrc || echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
grep -qxF 'export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH' ~/.bashrc || echo 'export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH' >> ~/.bashrc

# Clean up
rm ollama-linux-amd64.tgz
````

The server can be started by running:
```bash
ollama serve
````

Pull the model you want:
```bash
ollama pull "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q6_K"
````
### Running an Experiment

To run Experiment 23, execute the following command from the `src` folder:

```bash
python3 -m experiments.experiment_23.run_experiment_23      
````

### Running an experiment on a server

As the experiments do not require supervision, a way to make the exeriment run even when not using your own computer is to run it from a server.
Steps to doing this:

1. Clone the git project to the server

2. Download all training files into asr-llm in an NPSC folder

3. Run commands using tmux, nohup or window to let the program run even when current bash session is closed

