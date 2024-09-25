import json
import re
import requests


HOSTNAME = "129.241.113"
PORT = 8080
#url = f"http://{HOSTNAME}:{PORT}/completion"  # llama.cpp server
url = "http://129.241.113.29:8080/completion"
headers = {"Content-Type": "application/json"}

only_string_schema = {
    "type": "string"
}
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

def pred(
    instruction,
    max_tokens=1000,
    use_schema: str = "default",
    temp=0.0,  # temperature. 0: deterministic, 1+: random
    # min_p=0.1,  # minimum probability
    # max_p=0.9,  # maximum probability
    # top_p=0.9,  # nucleus sampling
    # top_k=40,  # consider top k tokens at each generation step
    evaluate: bool = False,  # apply eval
):
    if len(instruction) == 0:
        raise ValueError("Instruction cannot be empty")

    data = {
        "prompt": instruction,
        "n_predict": max_tokens,
        "temperature": temp,
        "repeat_penalty": 1.2,  # 1.1 default,
    }
    if use_schema:
        data["json_schema"] = schemas[use_schema]

    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    response = response["content"]
    if evaluate:
        return parse_llm_output(response)
    return response

def parse_llm_output(response: str):
    if not response:
        return response
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

    obj = eval(response)
    if isinstance(obj, dict):
        # unify keys in case of capitalization.
        obj = {k.lower(): v for k, v in obj.items()}
    return obj


def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(data)
    except FileNotFoundError:
        data = []  # If file doesn't exist, initialize with an empty list
    return data
    

def choose_best_sentence(context, choices):
    # Prepare the input for the LLM
    #prompt = "Given the context bellow, choose the most fitting sentence from the option set\n"
    prompt = "Gitt konteksten nedenfor, velg den mest passende setningen i alternativsettet nedenfor \n"
    prompt += f"Konteks: {' '.join(context)}\n"
    #prompt += "Kontekst: Solenergi er en av de raskest voksende kildene til fornybar energi, spesielt i områder med mye sollys.\n"
    
    prompt += f"\nAlternativ sett:\n"
    for j, choice in enumerate(choices):
        prompt += f"{j+1}: {choice}\n"
    # prompt += """ 
    # Alternativ sett: 
    # 1. "Det er mest effektivt i områder med mye regn."\n
    # 2. "Det brukes mest i regioner med sterk vind."\n
    # 3. "Det er mest effektivt på steder med mye sollys."\n
    # 4. "Det fungerer godt i kalde og mørke omgivelser."\n
    # 5. "Det er mest produktivt om vinteren."\n

    # """
    # examplar =  """
    # \n Context: nordmenn er nordlendinger trøndere sørlendinger og folk fra alle andre regioner
    # nordmenn er også innvandret fra afghanistan pakistan og polen sverige somalia og syria
    # det vi kaller hjem er der hjertet vårt er og det kan ikke alltid plasseres innenfor landegrenser
    # nordmenn er jenter som er glad i jenter gutter som er glad i gutter og jenter og gutter som er glad i hverandre
    # nordmenn tror på gud allahaltet og ingenting
    # nordmenn liker grieg hygo hygo \n
    # Option set: \n
    # 1. hellbillies og kari bremnes med andre ord norge er dere norge er oss mitt største håp for norge er at vi skal klare å ta vare på hverandre at vi skal bygge dette landet videre på tillit fellesskap og raushet \n 
    # 2. helbillies og kari bremnes med andre ord norge er dere norge er oss mitt største håp for norge er at vi skal klare å ta vare på hverandre at vi skal bygge dette landet videre på tillit fellesskap og raushet \n
    # 3. hellbillies og kari bremnes med andre ord norge er dere norge er oss mitt største håp for norge er at vi skal klare å ta vare på hverandre at vi skal bygge dette landet videre på tillit fellesskap og raushet \n
    # 4. hellbillies og kari bremnes med andre ord norge er dere norge er oss mitt største håp for norge er at vi skal klare å ta vare på hverandre at vi skal bygge dette landet videre på tillit fellesskap og raushet \n
    # 5. hellbillies og kari bremnes med andre ord norge er dere norge er oss mitt største håp for norge er at vi skal klare å ta vare på hverandre at vi skal bygge dette landet videre på tillit fellesskap og raushet \n

    # Output: hellbillies og kari bremnes med andre ord norge er dere norge er oss mitt største håp for norge er at vi skal klare å ta vare på hverandre at vi skal bygge dette landet videre på tillit fellesskap og raushet\n
    # """
    #prompt += 'Eksempel ut: String_field: " hellbillies og kari bremnes med andre ord norge er dere norge er oss mitt største håp for norge er at vi skal klare å ta vare på hverandre at vi skal bygge dette landet videre på tillit fellesskap og raushet", number_field: 2'

    # prompt += "Here is an example of expected input and output:" + examplar

    print("Prompt:", prompt)
    # prompt += "\nThe most suitable choice is:"
    print(pred(prompt))

    # # Define the payload
    # payload = {
    #     "prompt": prompt,
    #     "max_tokens": 100
    # }

    # # Make a POST request to the API
    # try:
    #     response = requests.post(url, json=payload)

    #     # Check if the request was successful
    #     if response.status_code == 200:
    #         # Print the response from the server
    #         print("Response:", response.json())
    #         return response.json()["content"]
    #     else:
    #         print(f"Request failed with status code {response.status_code}: {response.text}")
    # except requests.exceptions.RequestException as e:
    #     print(f"An error occurred: {e}")




data = read_file("result/result2.txt")
best_sentences = []    
for i, entry in enumerate(data):
        context = entry['context']
        choices = entry['choices']
        print(f"Processing entry {i + 1}...")
        best_choice = choose_best_sentence(context, choices)
        best_sentences.append(best_choice)
        print(f"Best choice for entry {i + 1}: {best_choice}")

print(best_sentences)