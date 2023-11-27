
import datasets
import openai
import pickle
import os
import re
import json
import torch
import time

def convert_dict_to_list(dataset_split):
    """
    returns list of samples for each feature
    """
    feature_list = list(dataset_split.features.keys())
    dataset_convert = dict()
    for feature in feature_list:
        dataset_convert[feature] = [dataset_split[feature][i] for i in range(len(dataset_split[feature]))]
    return dataset_convert


openai.api_key = os.getenv("OPENAI_API_KEY")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
data_name = "snarks"
ANS_PRMPT = {"snarks" : {"A1" : " The answer is "}}

# Load Dataset
dataset = datasets.load_dataset("tasksource/bigbench", 'snarks')
exp_test_data = dataset["validation"]
exp_test_data = convert_dict_to_list(exp_test_data)
exp_test_data["text"] = exp_test_data["inputs"]
exp_test_data["label"] = exp_test_data["targets"]


train_prompt = pickle.load(open("snarks_few_shot_prompt.pkl", "rb"))


# Create few-shot prompt for each test sample
encoded_test_samples = []
answers = []
for ind in range(len(exp_test_data["text"])):
    encoded_test_samples.append(
        train_prompt + exp_test_data["text"][ind].replace("\n", " "))
    answers.append(exp_test_data["label"][ind])


path_to_store = f"gpt3.5_turbo_responses_{data_name}_test.pkl"
responses = dict()
count_samples = 0
count_correct = 0
for ind, prompt in enumerate(encoded_test_samples):
    print(f"Processing {count_samples} ... ")
    count_samples += 1
    for _ in range(3):
        try:
            # response = ""
            response = openai.ChatCompletion.create(  # 1. Change the function Completion to ChatCompletion
                model='gpt-3.5-turbo-0301',
                messages=[  # 2. Change the prompt parameter to the messages parameter
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0,
                max_tokens=100,
            )
            break
        except openai.error.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(1)
        except json.decoder.JSONDecodeError as exception:
            print(f"JSON Error : {exception}. Retrying...")
            time.sleep(1)

    responses[ind] = response
    pickle.dump(responses, open(path_to_store, "wb"))
    response_token = response['choices'][0]['message']['content']
    regex_pattern = r"(?<=The answer is\s).*$"
    match = re.search(regex_pattern, response_token)
    if match == None:
        response_token = "None"
    else:
        match = match.group().split()
        match = "nothing" if len(match) == 0 else match[0]
        response_token = match.strip()[1]
    answer_token = answers[ind][0].lower()
    if response_token in answer_token:
        count_correct += 1

print("Accuracy : ", count_correct/count_samples)