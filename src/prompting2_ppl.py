from datetime import datetime
import os
import sys
import torch
from torch.utils.data import DataLoader
# import wandb
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import random
import csv
import re

device = torch.device('cuda:0')

common_prompt = """ 
    Your role is to assist in generating explanations for programming errors. You are designed to provide explanations for any type of programming error in exactly one sentence, no matter the complexity. Your goal is to encapsulate the essence of the error, its common cause, and a general solution in a concise manner. This requires distilling information into its most essential form, ensuring that each explanation is clear and directly addresses the error at hand. You aim to be accessible to beginners and informative for more experienced developers, all within a single sentence. 
    """
from datasets import load_dataset, concatenate_datasets

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


def tokenize(prompt):
    return tokenizer(
        prompt,
        # truncation=True,
        # max_length=256,
        # padding="max_length",  # ensure all prompts are the same length
        return_tensors='pt' # return PyTorch tensors
    )

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
    return tokenize(prompt_text)


# def generate_and_tokenize_prompt(data_point,common_prompt,):
#     full_prompt =f"""
#     ### Input:
#     {data_point["Error Text"]}

#     ### Response:
#     {data_point["Alignment"]}
#     """
#     full_prompt = common_prompt+full_prompt
    
#     return tokenize(full_prompt)


result_arr = []
if os.path.exists('./prompting_perplexity_test_set.csv'):
    with open('./prompting_perplexity_test_set.csv', mode='r', encoding='utf-8') as file:
    # Create a DictReader object
        csv_reader = csv.DictReader(file)
        
        # Append each row in the file to the list as a dictionary
        for row in csv_reader:
            result_arr.append(row)

random_seeds = [5]
for run_no,seed in enumerate(random_seeds):
   
    random.seed(seed)
    # random_indexes = random.sample(range(train_dataset.shape[0] + 1), 4) #choosing 4 random indexes between 0 and length of train set
    # for index in random_indexes:
    #     prompt_sample = train_dataset[index]
    #     prompt = f""" 
    #     ### Input:
    #     {prompt_sample['Error Text']}
        
    #     ### Response:
    #     {prompt_sample['Alignment']}
    #     """
    #     common_prompt = common_prompt + prompt

    tokenized_val_dataset = val_dataset.map(lambda err: prompt_ds(err, 4, common_prompt))
    # eval_prompt = prompt_ds(val_sample, 4)

    # val_dataloader = DataLoader(tokenized_val_dataset, batch_size=16)

    model_pretrained.eval()
    with torch.no_grad():
        total_loss = 0
        count = 0
        for batch in tokenized_val_dataset:  # This needs to be an actual DataLoader or 
    
            if isinstance(batch['input_ids'], list):
                batch['input_ids'] = torch.tensor(batch['input_ids'])
            if 'attention_mask' in batch and isinstance(batch['attention_mask'], list):
                batch['attention_mask'] = torch.tensor(batch['attention_mask'])

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()

            outputs = model_pretrained(input_ids=input_ids,attention_mask=attention_mask, labels=input_ids) 
            # Calculate the loss
            loss = outputs.loss
            print("loss",loss.item())
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        result_arr.append({'Prompting Strategy':'Semi-random (2)','Run':run_no,'random seed':seed,'Avg. Perplexity':round(perplexity,2)})
        print("Perplexity:", perplexity)
        
        with open(os.path.join('./prompting_perplexity_test_set.csv'), mode='w',newline='',encoding='utf-8') as csvfile:
																# create csv writer
            writer = csv.DictWriter(csvfile,fieldnames=result_arr[0].keys())
            writer.writeheader()
            # write data rows
            for row in result_arr:
                writer.writerow(row)