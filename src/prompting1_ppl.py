

from datetime import datetime
import os
import sys
import torch
from torch.utils.data import DataLoader
# import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import random
import csv

device = torch.device('cuda:0')
from datasets import load_dataset

train_dataset = load_dataset('json', data_files='./train.jsonl', split='train')
val_dataset = load_dataset('json', data_files='./val.jsonl', split='train')


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


def tokenize(prompt):
    return tokenizer(
        prompt,
        # truncation=True,
        # max_length=256,
        # padding="max_length",  # ensure all prompts are the same length
        return_tensors='pt' # return PyTorch tensors
    )

def generate_and_tokenize_prompt(data_point,common_prompt,):
    full_prompt =f"""
    ### Input:
    {data_point["Error Text"]}

    ### Response:
    {data_point["Alignment"]}
    """
    full_prompt = common_prompt+full_prompt
    
    return tokenize(full_prompt)


result_arr = []
if os.path.exists('./prompting_perplexity.csv'):
    with open('./prompting_perplexity.csv', mode='r', encoding='utf-8') as file:
    # Create a DictReader object
        csv_reader = csv.DictReader(file)
        
        # Append each row in the file to the list as a dictionary
        for row in csv_reader:
            result_arr.append(row)

random_seeds = [48,52,34,12,5]
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

    val_dataloader = DataLoader(tokenized_val_dataset, batch_size=16)

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

            # if attention_mask is not None:
            #     attention_mask = attention_mask.to(device)
            # print("label input id",label_input_ids.shape)
            # print(label_input_ids)
            # print("input id",input_ids.shape)
            # print(input_ids)
            # Generate outputs using the model
            outputs = model_pretrained(input_ids=input_ids,attention_mask=attention_mask, labels=input_ids) 
            # Calculate the loss
            loss = outputs.loss
            print("loss",loss.item())
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        result_arr.append({'Prompting Strategy':'Random (1)','Run':run_no,'random seed':seed,'Avg. Perplexity':round(perplexity,2)})
        print("Perplexity:", perplexity)

        with open(os.path.join('./prompting_perplexity.csv'), mode='w',newline='',encoding='utf-8') as csvfile:
																# create csv writer
            writer = csv.DictWriter(csvfile,fieldnames=result_arr[0].keys())
            writer.writeheader()
            # write data rows
            for row in result_arr:
                writer.writerow(row)