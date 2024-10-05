

from datetime import datetime
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
import re
from bert_score import score
import csv


import random
device = torch.device('cuda:0')
from datasets import load_dataset

train_dataset = load_dataset('json', data_files='./train.jsonl', split='train')
val_dataset = load_dataset('json', data_files='./test.jsonl', split='train')

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
def extract_response(model_output):

    # Regex to find the content after '###Response:' and before the next '###' or end of string
    matches = re.findall(r'### Response:\s*(.*?)\s*(?:###|$)', model_output, re.S)

    # Get the last match if there are any matches
    last_response = matches[-1] if matches else None

    return last_response

def bert_f1(y_pred_arr,y_gt_arr):

    model_type = "roberta-base"  # We can change the model based on the memory available

    # Calculate BERTScore using the specified model
    P, R, F1 = score(y_pred_arr,y_gt_arr, lang="en", model_type=model_type, verbose=True)

    return P,R,F1

# ### The below cell creates the common prompt with 4 input examples in the prompt chosen randomly with seed 48. The seed is changed for each of the 5 trials

result_arr = []
if os.path.exists('./prompting_results_test_set.csv'):
    with open('./prompting_results_test_set.csv', mode='r', encoding='utf-8') as file:
    # Create a DictReader object
        csv_reader = csv.DictReader(file)
        
        # Append each row in the file to the list as a dictionary
        for row in csv_reader:
            result_arr.append(row)

def tokenize(prompt):
    return tokenizer(
        prompt,
        # truncation=True,
        # max_length=256,
        # padding="max_length",  # ensure all prompts are the same length
        return_tensors='pt' # return PyTorch tensors
    )

def generate_and_tokenize_prompt(data_point,common_prompt):
    full_prompt =f"""
    ### Input:
    {data_point["Error Text"]}

    ### Response:
    """
    full_prompt = common_prompt+full_prompt
    
    return tokenize(full_prompt)

random_seeds = [5]
y_true_arr = val_dataset['Alignment']
for run_no,seed in enumerate(random_seeds):
    common_prompt = """ 
    Your role is to assist in generating explanations for programming errors. You are designed to provide explanations for any type of programming error in exactly one sentence, no matter the complexity. Your goal is to encapsulate the essence of the error, its common cause, and a general solution in a concise manner. This requires distilling information into its most essential form, ensuring that each explanation is clear and directly addresses the error at hand. You aim to be accessible to beginners and informative for more experienced developers, all within a single sentence. 
    """
    random.seed(seed)
    random_indexes = random.sample(range(train_dataset.shape[0] + 1), 4) #choosing 4 random indexes between 0 and length of train set
    for index in random_indexes:
        prompt_sample = train_dataset[index]
        prompt = f""" 
        ### Input:
        {prompt_sample['Error Text']}
        
        ### Response:
        {prompt_sample['Alignment']}
        """
        common_prompt = common_prompt + prompt
    
    tokenized_val_dataset = val_dataset.map(lambda x: generate_and_tokenize_prompt(x,common_prompt))

    y_pred_arr = []
    model_pretrained.eval()
    with torch.no_grad():
        for val_sample in tokenized_val_dataset:
            if isinstance(val_sample['input_ids'], list):
                val_sample['input_ids'] = torch.tensor(val_sample['input_ids'])

            if 'attention_mask' in val_sample and isinstance(val_sample['attention_mask'], list):
                val_sample['attention_mask'] = torch.tensor(val_sample['attention_mask'])
            input_ids =val_sample['input_ids'].to(device)
            attention_mask = val_sample['attention_mask'].to(device)
            model_output = tokenizer.decode(model_pretrained.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=70)[0])
            y_pred_arr.append(extract_response(model_output)) #just the ###Response part of the model output is considered the y_pred
        precision,recall,f1_scores = bert_f1(y_pred_arr,y_true_arr)
        avg_precision,avg_recall,avg_f1 = torch.mean(precision).item(),torch.mean(recall).item(),torch.mean(f1_scores).item()
        result_arr.append({'Prompting Strategy':'Random (1)','Run':run_no,'random seed':seed,'Avg. BERT score F1':round(avg_f1,2),'Avg. BERT score Precision':round(avg_precision,2),'Avg. BERT score Recall':round(avg_recall,2)})
        with open(os.path.join('./prompting_results_test_set.csv'), mode='w',newline='',encoding='utf-8') as csvfile:
            # create csv writer
            writer = csv.DictWriter(csvfile,fieldnames=result_arr[0].keys())
            writer.writeheader()
            # write data rows
            for row in result_arr:
                writer.writerow(row)