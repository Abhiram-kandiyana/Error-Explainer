#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
import re
from bert_score import score
import csv
from datasets import concatenate_datasets
import pandas as pd


import random

from datasets import load_dataset

train_dataset = load_dataset('json', data_files='./train.jsonl', split='train')
val_dataset = load_dataset('json', data_files='./test.jsonl', split='train')
random_indices = [24, 76, 65, 48, 36, 71, 50, 54, 12, 55]


# In[18]:


train_dataset.shape[0]


# ### Load model
# Loading the code-llama-7b stored in local

# In[7]:
common_prompt = """
Your role is to assist in generating explanations for programming errors. You are designed to provide explanations for any type of programming error in exactly one sentence, no matter the complexity. Your goal is to encapsulate the essence of the error, its common cause, and a general solution in a concise manner. This requires distilling information into its most essential form, ensuring that each explanation is clear and directly addresses the error at hand. You aim to be accessible to beginners and informative for more experienced developers, all within a single sentence.
"""

base_model = "codellama/CodeLlama-7b-hf" #The model is downloaded to the cache folder in home directory
model_pretrained = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
)

def extract_type(error_data):
    error = error_data['Error Text']
    if not isinstance(error, str):
        return "Input is not a string"
    match1 = re.search(r"\w+Error:", error)
    if match1:
        error_data['Error Type'] = match1.group(0)[:-1]
        return error_data
    match2 = re.search(r"\w+Error", error)
    if match2:
        error_data['Error Type'] = match2.group(0)
        return error_data
    match3 = re.search(r"\w+\.\w*error", error)
    if match3:
        error_data['Error Type'] = match3.group(0)
        return error_data
    error_data['Error Type'] = 'Error'
    return error_data

train_dataset = train_dataset.map(extract_type)
val_dataset = val_dataset.map(extract_type)
# test_dataset = test_dataset.map(extract_type)

def pick_errors_ds(train_dataset, error, k):
    error_type = error['Error Type'] # Assuming 'error_ds' is a Dataset object with one example
    # print('error_type',error_type)
    # Filter by 'Error Type'
    train_picked_ds = train_dataset.filter(lambda err: err['Error Type'] == error_type)

    if error_type == 'Error' or len(train_picked_ds) == 0:
        # Sample k entries
        sampled_indices = random.sample(range(len(train_dataset)), k) #np.random.choice(len(train_dataset), k, replace=False)
        return train_dataset.select(sampled_indices)
    elif len(train_picked_ds) < k:
        add_picked_ds = train_dataset.filter(lambda example: example['Error Type'] != error_type)
        needed = k - len(train_picked_ds)
        sampled_indices_add = random.sample(range(len(add_picked_ds)), needed) #np.random.choice(len(add_picked_ds), needed, replace=False)
        add_picked_ds = add_picked_ds.select(sampled_indices_add)
        # Concatenate Datasets
        combined_ds = concatenate_datasets([train_picked_ds, add_picked_ds])
        return combined_ds
    else:
        sampled_indices = random.sample(range(len(train_picked_ds)), k) #np.random.choice(len(train_picked_ds), k, replace=False)
        return train_picked_ds.select(sampled_indices)

def prompt_ds(test_error, k=4, desc = common_prompt, train_error_ds = train_dataset):
    prompt_text = desc
    if k > 0:
        prompt_errors_ds = pick_errors_ds(train_error_ds, test_error, k)

        # Create prompts for each error example in the picked errors dataset
        error_prompts = []
        for error in prompt_errors_ds:
            sub_prompt = f"""
                ### Input:
                {error['Error Text']}

                ### Response:
                {error['Alignment']}
            """
            error_prompts.append(sub_prompt)

        # Join all error prompts to the main prompt text
        prompt_text += "".join(error_prompts)

    # Assuming test_error_ds is a single example in a dataset
    # test_error = test_error  # Access the first (and expected only) record directly
    sub_prompt = f"""
        ### Input:
        {test_error['Error Text']}
        """
    prompt_text += sub_prompt
    return prompt_text


def extract_response(model_output):

    # Regex to find the content after '###Response:' and before the next '###' or end of string
    matches = re.findall(r'### Response:\s*(.*?)\s*(?:###|$)', model_output, re.S)

    # Get the last match if there are any matches
    last_response = matches[-1] if matches else None
    
    if len(last_response) < 15:
        last_response = matches[-2] if matches else last_response
    
    return last_response

def bert_f1(y_pred_arr,y_gt_arr):

    model_type = "roberta-base"  # We can change the model based on the memory available

    # Calculate BERTScore using the specified model
    P, R, F1 = score(y_pred_arr,y_gt_arr, lang="en", model_type=model_type, verbose=True)

    return P,R,F1

random_seeds = [5]
for run_no,seed in enumerate(random_seeds):

    y_pred_arr = []
    for index in random_indices:
        val_sample = val_dataset[index]
        eval_prompt = prompt_ds(val_sample, 4) #error data and K-shot as args

        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

        model_pretrained.eval()
        with torch.no_grad():
            model_output = tokenizer.decode(model_pretrained.generate(**model_input, max_new_tokens=70)[0])
            y_pred_arr.append(extract_response(model_output)) #just the ###Response part of the model output is considered the y_pred

    df = pd.read_excel('human_eval_outputs.xlsx')

    df['Prompting2 Output'] = y_pred_arr
    with pd.ExcelWriter('human_eval_outputs.xlsx', engine='openpyxl',mode='w') as writer:
        df.to_excel(writer, index=False)

    print("Excel file has been updated with the prompting2 output.")
        
