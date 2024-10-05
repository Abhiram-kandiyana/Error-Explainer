


from datetime import datetime
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
import re
from bert_score import score
import csv
from peft import PeftModel
import pandas as pd

import random

from datasets import load_dataset

device = torch.device('cuda:0')
test_dataset = load_dataset('json', data_files='./test.jsonl', split='train')
random_indices = [24, 76, 65, 48, 36, 71, 50, 54, 12, 55]


# ### Load model
# Loading the code-llama-7b stored in local


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

tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
#Download the finetuned model from "https://drive.google.com/drive/folders/125u-ebeP5bHsMXurBcN18AVuYHvYLYPu"  to this directory and save it as "code-llama-finetuned"
model_finetuned = PeftModel.from_pretrained(model_pretrained, "./code-llama-finetuned")

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

# ### The below cell creates the common prompt with 4 input examples in the prompt chosen randomly with seed 48. The seed is changed for each of the 5 trials

# In[72]:



result_arr = []
if os.path.exists('./prompting_results_test_set_1.csv'):
    with open('./prompting_results_test_set_1.csv', mode='r', encoding='utf-8') as file:
    # Create a DictReader object
        csv_reader = csv.DictReader(file)
        
        # Append each row in the file to the list as a dictionary
        for row in csv_reader:
            result_arr.append(row)


common_prompt = """ 
Your role is to assist in generating explanations for programming errors. You are designed to provide explanations for any type of programming error in exactly one sentence, no matter the complexity. Your goal is to encapsulate the essence of the error, its common cause, and a general solution in a concise manner. This requires distilling information into its most essential form, ensuring that each explanation is clear and directly addresses the error at hand. You aim to be accessible to beginners and informative for more experienced developers, all within a single sentence. 
"""


y_pred_arr = []
y_true_arr = []
for index in random_indices:
    
    # print(y_true)
    # print("*"*50)
    test_sample = test_dataset[index]
    eval_prompt = f"""
    ### Input:
    {test_sample['Error Text']}
    
    ### Response:
    """
    eval_prompt = common_prompt + eval_prompt
    # print(eval_prompt)
    # print("*"*50)
    
    model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)

    model_finetuned.eval()
    with torch.no_grad():
        model_output = tokenizer.decode(model_finetuned.generate(**model_input, max_new_tokens=70)[0])
        y_pred_arr.append(extract_response(model_output)) #just the ###Response part of the model output is considered the y_pred

    y_true_arr.append(test_sample['Alignment'])
df = pd.read_excel('human_eval_outputs.xlsx')

df['Finetuned Output'] = y_pred_arr
df['GT(Human + GPT-4)'] = y_true_arr
with pd.ExcelWriter('human_eval_outputs.xlsx', engine='openpyxl',mode='w') as writer:
    df.to_excel(writer, index=False)

print("Excel file has been updated with the finetuned output.")
