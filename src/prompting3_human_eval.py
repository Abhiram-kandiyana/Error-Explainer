


from datetime import datetime
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
import re
from bert_score import score
import csv
import pandas as pd
import random

from datasets import load_dataset

train_dataset = load_dataset('json', data_files='./train.jsonl', split='train')
val_dataset = load_dataset('json', data_files='./test.jsonl', split='train')
random_indices = [24, 76, 65, 48, 36, 71, 50, 54, 12, 55]



# ### Load model
# Loading the code-llama-7b stored in local

# In[7]:


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


def extract_response(model_output):

    # Regex to find the content after '###Response:' and before the next '###' or end of string
    matches = re.findall(r'### Response:\s*(.*?)\s*(?:###|$)', model_output, re.S)

    # Get the last match if there are any matches
    last_response = matches[-1] if matches else None
    
    if len(last_response) < 15:
        last_response = matches[-2] if matches else last_response
    
    return last_response

# ### The below cell creates the common prompt with 4 input examples in the prompt chosen randomly with seed 48. The seed is changed for each of the 5 trials

# In[72]:



#{AttributeError, FileNotFindError, CppError, RuntimeError} sets manually chosen to increase breadth, or 'coverage', of prompting examples
#the idea for this strategy is to select sets of dissimilar error texts regardless of the error that model is to explain
training_error_indeces = [[265,529,32,8]] #changed
for run_no,error_indexes in enumerate(training_error_indeces):
    common_prompt = """
    Your role is to assist in generating explanations for programming errors. You are designed to provide explanations for any type of programming error in exactly one sentence, no matter the complexity. Your goal is to encapsulate the essence of the error, its common cause, and a general solution in a concise manner. This requires distilling information into its most essential form, ensuring that each explanation is clear and directly addresses the error at hand. You aim to be accessible to beginners and informative for more experienced developers, all within a single sentence.
    """ 
    for index in error_indexes:
        prompt_sample = train_dataset[index]
        prompt = f""" 
        ### Input:
        {prompt_sample['Error Text']}
        
        ### Response:
        {prompt_sample['Alignment']}
        """
        common_prompt = common_prompt + prompt

    y_pred_arr = []
    y_true_arr = []
    for index in random_indices:
        
        # print(y_true)
        # print("*"*50)
        val_sample = val_dataset[index]
        eval_prompt = f"""
        ### Input:
        {val_sample['Error Text']}
        
        ### Response:
        """
        eval_prompt = common_prompt + eval_prompt
        
        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

        model_pretrained.eval()
        with torch.no_grad():
            model_output = tokenizer.decode(model_pretrained.generate(**model_input, max_new_tokens=70)[0])
            y_pred_arr.append(extract_response(model_output)) #just the ###Response part of the model output is considered the y_pred
        y_true_arr.append(val_sample['Alignment'])

    df = pd.read_excel('human_eval_outputs.xlsx')

    df['Prompting3 Output'] = y_pred_arr
    with pd.ExcelWriter('human_eval_outputs.xlsx', engine='openpyxl',mode='w') as writer:
        df.to_excel(writer, index=False)

    print("Excel file has been updated with the prompting3 outputs.")

    