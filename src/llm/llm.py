## This is called directly into the Whisper model ##

import json
import os
import re
import requests
from dotenv import load_dotenv
load_dotenv() 

HOSTNAME = os.environ["HOSTNAME"]
PORT = os.environ["PORT"]
headers = {"Content-Type": "application/json"}
 
only_string_schema = {
    "type": "string"
}

only_string_schema_object = {
    "type": "object",
    "properties": {
        "result":{
            "type": "text"
        }
    }
}

#TODO Add object

#TODO Test schema that have to use one of the
only_number_schema = {
    "type": "number",
    "enum": [1, 2, 3, 4, 5]
}

both_schema = {
    "type": "object",
    "properties": {
        "string_field": {
            "type": "string"
        },
        "number_field": {
            "type": "number",
            "enum": [1, 2, 3, 4, 5]
        }
    },
    "required": ["string_field", "number_field"]
}

schemas = {
    "default": only_string_schema,
    "number_only": only_number_schema,
    "both": both_schema
}

def returnLogits():
    model = Llama(model_path="../../llama.cpp/models/gemma-2-9b-it-Q6_K_L.gguf?download=true", logits_all=True)
    out = model.create_completion("The capital of France is", max_tokens=1, logprobs=20)
    print(out)

def pred(
    instruction,
    max_tokens=1000,
    use_schema: str = "default",
    temp=0.1,  # temperature. 0: deterministic, 1+: random
    # min_p=0.1,  # minimum probability
    # max_p=0.9,  # maximum probability
    # top_p=0.9,  # nucleus sampling
    # top_k=40,  # consider top k tokens at each generation step
    context = [],
    choices = [],
    evaluate: bool = False,  # apply eval
    iteration: int= 0,
    port=8081
):
    if len(instruction) == 0:
        raise ValueError("Instruction cannot be empty")

    data = {
        "prompt": instruction,
        "n_predict": max_tokens,
        "temperature": temp,
        "repeat_penalty": 1.2,  # 1.1 default,
        "logits_all": True
    }
    if use_schema:
        data["json_schema"] = schemas[use_schema]
    print("poooooort", port)
    url = f"http://{HOSTNAME}:{port}/completion"  # llama.cpp server
    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    print(response)
    response = response["content"]
    if evaluate:
        return parse_llm_output(response)
    return response

def parse_llm_output(response: str):
    if not response:
        return response
    print("response", response)
    # spacing!
    response = response.replace("\n", " ")
    response = response.replace("\t", " ")
    response = re.sub(r"\s+", " ", response)
    response = response.strip()
    # markdown ticks
    response = response.replace("```python", "")
    response = response.replace("```json", "")
    response = response.replace("```", "")

    response = response.replace("false", "False")
    response = response.replace("true", "True")
    response = response.replace("null", "None")
    response = response.replace("The selected top1 ASR transcription", "")
    response = response.replace("The selected top-1 ASR transcription", "")
    response = response.replace("Selected Transcription:", "")
    response = response.replace("Selected Transcription", "")

    #response = response.lower()

    # Regular expression to capture text between <optionnumber> tags
    match = re.search(r'<option\d+>(.*?)</option\d+>',response)


    # Remove any HTML code still left
    CLEANR = re.compile('<.*?>')
    if match:
        extracted_text = match.group(1)
        print("extracted", extracted_text)
        cleantext = re.sub(CLEANR, '', extracted_text)
        print("cleannn", cleantext.strip())
        cleantext = re.sub(r'^.*?:', '', cleantext)
        if('<option1>' in cleantext):
            raise Exception("Parsing error")
        if ('<option2>' in cleantext):
            raise Exception("Parsing error")
        if('<option3>' in cleantext):
            raise Exception("Parsing error")
        if('<option4>' in cleantext):
            raise Exception("Parsing error")
        if('<option5>' in cleantext):
            raise Exception("Parsing error")
        if('print(' in cleantext):
            raise Exception("Parsing error")
        if(not cleantext):
             raise Exception("Empty string")
        return cleantext.strip()
    else:
        raise Exception("No option returned")


def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []  # If file doesn't exist, initialize with an empty list
    return data

def try_streaming_output():
    headers = {"Content-Type": "application/json"}
    url = f"http://{HOSTNAME}:{8082}/completion"

    data = {
        "prompt": "Building a website can be done in 10 simple steps",
        "n_predict":1000,
        "temperature": 0.1,
        "repeat_penalty": 1.2,  # 1.1 default,
        "logits_all": True,
        "n_probs": 100
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print(f"Error: Received HTTP {response.status_code}")
        #print("Response Text:", response.text)  # Print raw response
    else:
        try:
            json_data = response.json()
            print("check out", json_data["completion_probabilities"])
            for probability in json_data["completion_probabilities"]:
                print("probability", probability)
        except requests.exceptions.JSONDecodeError:
            print("Error: Response is not valid JSON")
            print("Raw response:", response.text)
import llama_cpp
def model_logits():
    model = llama_cpp.Llama(model_path="../../llama.cpp/models/gemma-2-9b-it-Q6_K_L.gguf?download=true", logits_all=True)
    out = model.create_completion("The capital of France is", max_tokens=1, logprobs=20)
    print(out)
            

def choose_best_sentence(context, choices, port=8081):
    # prompt = "You are an ASR transcript selector." 
    # prompt += "Perform language model rescoring based on the top5 outputs generated by an Automatic Speech Recognition (ASR) system given a conversational history.\n"
    # prompt += "Select the most accurate ASR transcription\n"
    # prompt +=f"History:  [{', '.join(context)}]\n\n"
    # prompt += "The ASR hypotheses are as follows:\n"
    # for i, choice in enumerate(choices, 1):
    #     prompt += f"<option{i}> {choice} </option{i}>\n"

    # prompt += (
    #  "Output the selected top-1 ASR transcription in the format: <optionX> The selected top1 ASR transcription </optionX>\n"
    #  "Do NOT include your reasoning."
    # )

    prompt = "<|start_header_id|>system<|end_header_id|>\n"
    prompt += "You are an ASR transcript selector.\n"
    prompt += "Perform language model rescoring based on the top5 outputs generated by an Automatic Speech Recognition (ASR) system given a conversational history.\n"
    prompt += "Select the most accurate ASR transcription.\n<|eot_id|>\n"

    prompt += "<|start_header_id|>user<|end_header_id|>\n"
    prompt += f"History: [{', '.join(context)}]\n\n"
    prompt += "The ASR hypotheses are as follows:\n"

    for i, choice in enumerate(choices, 1):
        prompt += f"<option{i}> {choice} </option{i}>\n"

    prompt += (
        "\nOutput the selected top-1 ASR transcription in the format: <optionX> The selected top1 ASR transcription </optionX>\n"
        "Do NOT include your reasoning.<|eot_id|>\n"
    )

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"



    
    print("Prompt:", prompt)
    return pred(prompt, context=context, choices=choices, evaluate=True, port=port)


# context = [
#             "what causes it that the government party right is not more concerned about these vulnerable children here",
#             "stensham",
#             "president jeg vil f\u00f8rst ta avstand fra premisset for de sp\u00f8rsm\u00e5lene",
#             "vi er sannsynligvis akkurat like opptatt av disse barna og disse m\u00f8drene",
#             "\"><option1> ```python import re from typing import List, Dict, Tuple def select_best_transcript(hypotheses: List[str], history: str) -> str:  # TODO: Implement your ASR transcript selection logic here  # You can use the provided hypotheses and history to determine the most accurate transcription.  # For example, you could consider factors like:  # - Semantic coherence with the conversation history  # - Pronunciation accuracy  # - Confidence scores from the ASR system (if available)  raise NotImplementedError(\"\n",
#             "det er ikke s\u00e5nn at alle fagfolk har v\u00e6rt enige om hva som var riktig i det sp\u00f8rsm\u00e5let her",
#             "\"><option1> ```python import re from typing import List, Dict, Tuple def select_best_transcript(hypotheses: List[str], history: str) -> str:  # TODO: Implement your ASR transcript selection logic here  # You can use the provided hypotheses and history to determine the most accurate transcription.  # For example, you could consider factors like:  # - Semantic coherence with the conversation history  # - Pronunciation accuracy  # - Confidence scores from the ASR system (if available)  raise NotImplementedError(\"\n",
#             "\"><option1> ```python import re from typing import List, Dict, Tuple def select_best_transcript(hypotheses: List[str], history: str) -> str:  # TODO: Implement your ASR transcript selection logic here  # You can use the provided hypotheses and history to determine the most accurate transcription.  # For example, you could consider factors like:  # - Semantic coherence with the conversation history  # - Pronunciation accuracy  # - Confidence scores from the ASR system (if available)  raise NotImplementedError(\"\n",
#             "\"><option1> ```python import re from typing import List, Dict, Tuple def select_best_transcript(hypotheses: List[str], history: str) -> str:  # TODO: Implement your ASR transcript selection logic here  # You can use the provided hypotheses and history to determine the most accurate transcription.  # For example, you could consider factors like:  # - Semantic coherence with the conversation history  # - Pronunciation accuracy  # - Confidence scores from the ASR system (if available)  raise NotImplementedError(\"\n"
#         ]

# choices = [ "neste replikant er nicholas wilkinson deretter ketil kjenseth",
#             "neste replikant er nicholas wilkinson deretter ketil kjenseth",
#             "neste replikant er nicholas wilkinson deretter ketil kjenseth",
#             "neste replikant er nicholas wilkinson deretter ketil kjenseth",
#             "neste replikant er nicholas wilkinson deretter ketil kjenseth"]

# predicted = choose_best_sentence(context, choices)
# print("preeeed", predicted)

# parsed = parse_llm_output("<option1><option1> ```python import re from typing import List, Dict, Tuple def select_best_transcript(hypotheses: List[str], history: str) -> str:  # TODO: Implement your ASR transcript selection logic here  # You can use the provided hypotheses and history to determine the most accurate transcription.  # For example, you could consider factors like:  # - Semantic coherence with the conversation history  # - Pronunciation accuracy  # - Confidence scores from the ASR system (if available)  raise NotImplementedError(\"\n</option1>")
# print("parsed", parsed)