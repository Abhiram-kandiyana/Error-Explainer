Requirements:

    Please run "pip install -r requirements.txt" in terminal before running any code to install all the required packages.

    Please change the path to the input and output files (.jsonl, .csv) to the correct directories before running the files. We have set all the paths to relative paths from the python files.


This directory has the source code for the following methods impemented in our project:

    1. ZSP: Evaluating the code-llama model directly without incorporating any prompt strategy or finetuning.
    2. finetuned: Evaluating the finetuned code-llama model. The finetuned model is stored at https://drive.google.com/drive/folders/125u-ebeP5bHsMXurBcN18AVuYHvYLYPu"  to this directory and save it as "code-llama-finetuned" and now you can run the finetuned_bertscore, finetuned_human_eval and finetuned_perplexity python files.
    3. prompting-1: Evaluating the code-llama model with random-sample-4-shot prompting. More details in the report.
    4. prompting-2: Evaluating the code-llama model with Same-class-sample-4-shot prompting. More details in the report.
    5. prompting-3: Evaluating the code-llama model with Deterministic 4-shot prompting. More details in the report.

The below is the description of the python files and when should you run it:

    1. The .py files ending with _bertscore calculate the BERT F1 score for the corresponding method mentioned in the name. For example, prompting1_bertscore calculates BERT score for random-sample 4 shot prompting.
    2. The .py files ending with _ppl calculate the perplexity score for the corresponding method mentioned in the name. For example, prompting1_perplexity calculates perplexity score for random-sample 4 shot prompting.
    3. The .py files ending with _human_eval save the outputs of the method into "human_eval_outputs.xlsx" file. They donot calculate the human eval score directly. For example, prompting1_human_eval adds the random-sample 4 shot prompting outputs to the human_eval_outputs.xlsx file.
    4. For finetuning the model, it is preferred to make a copy and run the google colab notebook at https://colab.research.google.com/drive/1lrHnpoabH6KwR9QlOO2tY0YGhCGMtW0T?usp=sharing. Otherwise you can run the finetuning_pretrained.ipynb



