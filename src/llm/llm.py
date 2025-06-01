## This is called directly into the Whisper model ##

import json
import os
import re
import requests
from dotenv import load_dotenv
load_dotenv() 

from typing import Optional

# TODO Using ollama, change if something else should be used
from ollama import ChatResponse, chat
from pydantic import BaseModel
from pydantic.types import JsonSchemaValue, Literal

model = "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q6_K"


# Classes used for Controlled decoding

class HypothesisSelector(BaseModel):
    selected:  Literal[1, 2, 3, 4, 5]

class HypothesisSelectorReasoning(BaseModel):
    reason: str
    selected:  Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

class TranscriptionCreater(BaseModel):
    transcription: str

class TranscriptionCreaterReasoning(BaseModel):
    reason: str
    transcription: str

class SummaryCreator(BaseModel):
    summary: str

class SummaryReasonCreator(BaseModel):
    reason: str
    summary: str

HOSTNAME = os.environ["HOSTNAME"]
PORT = os.environ["PORT"]
headers = {"Content-Type": "application/json"}


def pred_summary(
    instruction,
    max_tokens=1000,
    use_schema: str = "default",
    temp=0.1,  # temperature. 0: deterministic, 1+: random
    num_ctx: int =48000,
    # min_p=0.1,  # minimum probability
    # max_p=0.9,  # maximum probability
    # top_p=0.9,  # nucleus sampling
    # top_k=40,  # consider top k tokens at each generation step
    iteration: int= 0,
    port=8081
):
    if len(instruction) == 0:
        raise ValueError("Instruction cannot be empty")

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
            format=SummaryReasonCreator.model_json_schema(),
        )

    response = response.message.content
    try:
        parsed_response = json.loads(response)  
        json_output = json.dumps(parsed_response, indent=4)
        print(json_output)
        summary = parsed_response.get("summary")
        summary = parse_llm_output(summary)
        reason = parsed_response.get("reason")
        print("\nReason:", reason)
            # Convert to JSON format
    except:
        raise("Could not parse json output")
    return summary

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
            format=TranscriptionCreater.model_json_schema(),
        )

        response = response.message.content
        print("response", response)
        try:
            parsed_response = json.loads(response)  
            json_output = json.dumps(parsed_response, indent=4)
            print(json_output)
            #TODO: if reasoning

            #reason = parsed_response.get("reason")
            
            # TODO: change when used as error correcto
            transcription = parsed_response.get("transcription")
            transcription = parse_llm_output(transcription)

            #index_chosen = parsed_response.get("selected")
            #print("index", index_chosen)
            #transcription = choices[index_chosen-1]
            # Convert to JSON format
        except:
            if(len(set(choices))) == 1:
                transcription = choices[0].replace("<|notimestamps|>", "")
                reason=""
            else:
                raise("Could not parse json output")
    else:
        selected_hypothesis=choices[0]
        selected_index = 0
        reason = ""
        transcription = ""

    print(transcription)
    # TODO remove reason if not included
    return transcription

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
    response = re.sub(r"[\u200B-\udfff]", "", response)
    response =  response.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')
    

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
        cleantext = re.sub(r'<s>[^<]*</s>', '', response)
        cleantext = re.sub(CLEANR, '', cleantext)
        # CLEANR_2 = re.compile(r'<[^>]+>')  
        # cleantext = re.sub(CLEANR_2, "", cleantext)
        cleantext = re.sub(r'^.*?:', '', cleantext)
        print(cleantext)
        
    return cleantext.strip()

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

def creating_summary(context,port=8081):
    prompt_asr = """
    You are a summary creator.
    Create a concise summary of the previous sentences generated by an ASR (Automatic Speech Recognition) system.
    Focus on the main idea, removing unnecessary filler and side comments.
    Preserve essential information such as names, dates, locations, and numerical values.
    ___
    History:
    {history}
    ___

    Reason concisely about the best summary of the past conversational history, and output the summary as a plain string.
    
    Always answer in Norwegian
    """
     
    prompt = prompt_asr.format(
        history=context,
    )

    return pred_summary(prompt, port=port)


            

def choose_best_sentence(context, choices, port=8081):

    # TODO When summary is created
    # creating_summary(context)

    #___New_prompt 
    prompt_asr = """
    You are an ASR transcript selector.
    Perform language model rescoring based on the top-5 outputs generated by an Automatic Speech Recognition (ASR) system given a conversational history,
    Select the most accurate ASR transcription and provide the best possible corrected version if necessary.    
    ___
    History:
    {history}
    ___
    The ASR hypotheses are as follows:
    {hypotheses}
    ___

    

    Output the best transcription as a plain string, making minimal corrections if necessary for accuracy and fluency, ensuring it follows the larger conversational history.
    Always answer in Norwegian. 
    """

    history_str = "\n\n".join(context)
    hypo = [f"<option {i+1}> {h.strip()} </option {i+1}" for i, h in enumerate(choices)]
    hypo_str = "\n".join(hypo)
    prompt = prompt_asr.format(
        history=context,
        hypotheses=hypo_str,
    )
    return pred(prompt, context=context, choices=choices, evaluate=True, port=port)


