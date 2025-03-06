## This is called directly into the Whisper model ##

import json
import os
import re
import requests
from dotenv import load_dotenv
load_dotenv() 

from typing import Optional

from ollama import ChatResponse, chat
from pydantic import BaseModel
from pydantic.types import JsonSchemaValue, Literal

model = "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q6_K"

class HypothesisSelector(BaseModel):
    selected:  Literal[1, 2, 3, 4, 5]

class HypothesisSelectorReasoning(BaseModel):
    reason: str
    selected:  Literal[1, 2, 3, 4, 5]

class TranscriptionCreater(BaseModel):
    transcription: str

class TranscriptionCreaterReasoning(BaseModel):
    reason: str
    transcription: str

HOSTNAME = os.environ["HOSTNAME"]
PORT = os.environ["PORT"]
headers = {"Content-Type": "application/json"}
 
def returnLogits():
    model = Llama(model_path="../../llama.cpp/models/gemma-2-9b-it-Q6_K_L.gguf?download=true", logits_all=True)
    out = model.create_completion("The capital of France is", max_tokens=1, logprobs=20)
    print(out)


def generate(
    # system_prompt: str,  # prøv uten først.
    prompt: str,
    schema: Optional[JsonSchemaValue] = None,
    parse: bool = True,
    num_ctx: int = 48000,
    num_predict: int = 4000,
    temperature: float = 0.0,
) -> str:
    response: ChatResponse = chat(
        model=model,
        messages=[
            # {"role": "system", "content": system_prompt},
            prompt,
        ],
        options={
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_k": 100,
            "top_p": 0.8,
            "temperature": temperature,
            "seed": 0,  # this is not needed when temp is 0
            "repeat_penalty": 1.3,  # remain default for json outputs, from experience.
        },
        format=schema,
        stream=False,
    )
    res = response.message.content
    if parse and schema:
        try:
            res = eval(res)
        except Exception:
            res = None
    return res


def pred(
    instruction,
    max_tokens=1000,
    use_schema: str = "default",
    temp=0.1,  # temperature. 0: deterministic, 1+: random
    num_ctx: int =48000,
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
    
    print("instruction", instruction)
    if(len(choices)>1):
        response: ChatResponse = chat(
            model=model,
            messages=[
                # {"role": "system", "content": system_prompt},
            {"role":"user","content":instruction}
            ],
            options={
                "num_ctx": num_ctx,
                "num_predict": max_tokens,
                "top_k": 100,
                "top_p": 0.8,
                "temperature": temp,
                "seed": 0,  # this is not needed when temp is 0
                "repeat_penalty": 1.3,  # remain default for json outputs, from experience.
            },
            stream=False,
            format=TranscriptionCreaterReasoning.model_json_schema(),
        )
        print("response 1", response)

        response = response.message.content
        print("response", response)
        try:
            parsed_response = json.loads(response)  
            json_output = json.dumps(parsed_response, indent=4)
            print(json_output)
            reason = parsed_response.get("reason")
            transcription = parsed_response.get("transcription")
            transcription = parse_llm_output(transcription)
            # Convert to JSON format
        except:
            if(len(set(choices))) == 1:
                transcription = choices[0].replace("<|notimestamps|>", "")
            else:
                raise("Could not parse json output")

        # selected_index = parsed_response.get("selected") 
        # reason = parsed_response.get("reason") 
        # print(selected_index)

        # if selected_index is not None and 0 < selected_index <= len(choices):
        #     selected_hypothesis = choices[selected_index - 1].replace("<|notimestamps|>", "")
        # else:
        #     raise Exception("Could not parse output")
    else:
        selected_hypothesis=choices[0]
        selected_index = 0
        reason = ""
        transcription = ""

    print(transcription)
    # if evaluate:
    #     return parse_llm_output(response)
    return transcription, reason

def parse_llm_output(response: str):
    if not response:
        return response
    print("\n\n -------------- PARSING -----------------")
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

    # Removing all invisible code
    response = re.sub(r"[\u200B-\u200D]", "", response)

    response = response.lower()

    # Regular expression to capture text between <optionnumber> tags
    match = re.search(r'<option\d+>(.*?)</option\d+>',response)


    # Remove any HTML code still left
    CLEANR = re.compile('<.*?>')
    if match:
        extracted_text = match.group(1)
        print("extracted", extracted_text)
        cleantext = re.sub(CLEANR, '', extracted_text)
        # CLEANR_2 = re.compile(r'<[^>]+>')  
        # cleantext = re.sub(CLEANR_2, "", cleantext)
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
        print("response", response)
        cleantext = re.sub(r'<s>[^<]*</s>', '', response)
        cleantext = re.sub(CLEANR, '', cleantext)
        # CLEANR_2 = re.compile(r'<[^>]+>')  
        # cleantext = re.sub(CLEANR_2, "", cleantext)
        cleantext = re.sub(r'^.*?:', '', cleantext)
        print(cleantext)
        
    return cleantext.strip()


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

    messages = [
      {
            "role": "system",
            "content": """You are an ASR transcript selector.\n
            Perform language model rescoring based on the top5 outputs generated by an Automatic Speech Recognition (ASR) system given a conversational history.\n
            Select the most accurate ASR transcription..""",
      },
      {
            "role": "user",
            "content": "Write a python function to generate the nth fibonacci number.",
      }
]


    #___New_prompt 
    prompt_asr = """
    You are an ASR transcript selector.
    Perform language model rescoring based on the top-5 outputs generated by an Automatic Speech Recognition (ASR) system given a conversational history.
    Select the most accurate ASR transcription and provide the best possible corrected version if necessary.
    ___
    History:
    {history}
    ___

    The ASR hypotheses are as follows:
    {hypotheses}
    ___

    Follow these steps to determine the correct transcription:
    - Analyze the hypotheses and conversational history to determine the most contextually and grammatically accurate transcription.  
    - Make only minimal corrections when necessary to ensure accuracy and fluency.  
    - Avoid repetition of previous statements or getting stuck on the same sentence structure.
    - Do not use emojis in your response.

    Output a concise reasoning for the best ASR transcription following the steps outlined above, and the final transcription as plaintext. 

    Always answer in Norwegian.
    """

    history_str = "\n\n".join(context)
    hypo = [f"Option {i+1} {h.strip()}" for i, h in enumerate(choices)]
    hypo_str = "\n".join(hypo)

##TODO test with history_str instead
    prompt = prompt_asr.format(
        history=context,
        hypotheses=hypo_str,
    )
    print(prompt)


    # class HypothesisSelector(BaseModel):
    #     selected: str


    # print(
    #     generate(
    #         prompt=prompt,
    #         schema=HypothesisSelector.model_json_schema(),
    #     )
    # )
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

from typing import Optional

from ollama import ChatResponse, chat
from pydantic import BaseModel
from pydantic.types import JsonSchemaValue, Literal

model = "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q6_K"
#model = "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Q6_K"


def generate(
    # system_prompt: str,  # prøv uten først.
    prompt: str,
    schema: Optional[JsonSchemaValue] = None,
    parse: bool = True,
    num_ctx: int = 48000,
    num_predict: int = 4000,
    temperature: float = 0.0,
) -> str:
    response: ChatResponse = chat(
        model=model,
        messages=[
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        options={
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_k": 100,
            "top_p": 0.8,
            "temperature": temperature,
            "seed": 0,  # this is not needed when temp is 0
            "repeat_penalty": 1.3,  # remain default for json outputs, from experience.
        },
        format=schema,
        stream=False,
    )
    res = response.message.content
    if parse and schema:
        try:
            res = eval(res)
        except Exception:
            res = None
    return res

